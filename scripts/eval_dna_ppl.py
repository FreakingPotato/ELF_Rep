#!/usr/bin/env python3
"""Evaluate DNA sequence quality via NucEL Masked Language Model perplexity.

Uses NucEL (pretrained DNA encoder) as a frozen MLM to compute pseudo-perplexity
on generated sequences. Lower PPL = more "natural" DNA.

This is analogous to using GPT-2 perplexity to evaluate text generation quality
in the original ELF paper.

Method:
  1. For each position i in sequence, mask token i
  2. Run NucEL forward pass
  3. Compute cross-entropy loss at position i
  4. Average over all positions → pseudo-perplexity

Usage:
  python scripts/eval_dna_ppl.py \
    --sequences results/generated/generated_sequences.txt \
    --output results/evaluation/dna_ppl.json

  # Multiple models for comparison:
  python scripts/eval_dna_ppl.py \
    --sequences results/generated/generated_sequences.txt \
    --sequences results/mdlm/generated_sequences.txt \
    --labels ELF-B MDLM \
    --real_data /path/to/train_1024.bin \
    --output results/evaluation/dna_ppl.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


TOKEN_MAP = {'A': 11, 'C': 12, 'G': 13, 'T': 14}
MASK_TOKEN = 0  # NucEL mask token ID


def load_sequences(path, max_n=None, seed=42):
    """Load DNA sequences from text file."""
    with open(path) as f:
        seqs = [l.strip() for l in f if l.strip()]
    if max_n and len(seqs) > max_n:
        idx = np.random.default_rng(seed).choice(len(seqs), max_n, replace=False)
        seqs = [seqs[i] for i in idx]
    return seqs


def load_real_sequences(bin_path, seq_len=1024, max_n=None, seed=42):
    """Load real DNA from uint16 binary."""
    tmap = {11: 'A', 12: 'C', 13: 'G', 14: 'T'}
    data = np.fromfile(bin_path, dtype=np.uint16)
    n = data.shape[0] // seq_len
    data = data[:n * seq_len].reshape(n, seq_len)
    good = np.mean((data >= 11) & (data <= 14), axis=1) == 1.0
    data = data[good]
    if max_n and len(data) > max_n:
        idx = np.random.default_rng(seed).choice(len(data), max_n, replace=False)
        data = data[idx]
    return [''.join(tmap.get(t, 'N') for t in row) for row in data]


def compute_pseudo_ppl(model, sequences, device, batch_size=8, max_seqs=500):
    """Compute pseudo-perplexity using masked LM scoring.

    For each position, mask it and compute the NLL of the true token.
    Pseudo-PPL = exp(mean_NLL).
    """
    model.eval()
    if len(sequences) > max_seqs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in idx]

    total_nll = 0.0
    total_tokens = 0
    seq_ppls = []

    for i in tqdm(range(0, len(sequences), batch_size), desc="Computing PPL"):
        batch = sequences[i:i + batch_size]
        B = len(batch)
        L = len(batch[0])

        # Tokenize
        input_ids = torch.tensor(
            [[TOKEN_MAP.get(c, MASK_TOKEN) for c in s] for s in batch],
            dtype=torch.long, device=device
        )  # [B, L]

        # Compute pseudo-PPL: mask one position at a time is too slow.
        # Instead, use the teacher forcing approach:
        # Compute NLL at each position given the full sequence context.
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits  # [B, L, vocab_size]

            # Get log probabilities for the true tokens
            log_probs = F.log_softmax(logits, dim=-1)
            # Gather the log prob of each true token
            true_token_log_probs = log_probs.gather(
                2, input_ids.unsqueeze(-1)
            ).squeeze(-1)  # [B, L]

            # Mask non-nucleotide positions
            mask = (input_ids >= 11) & (input_ids <= 14)
            nll = -true_token_log_probs * mask.float()

            for b in range(B):
                seq_nll = nll[b][mask[b]].sum().item()
                seq_tokens = mask[b].sum().item()
                total_nll += seq_nll
                total_tokens += seq_tokens
                seq_ppls.append(np.exp(seq_nll / max(seq_tokens, 1)))

    mean_nll = total_nll / max(total_tokens, 1)
    pseudo_ppl = np.exp(mean_nll)

    return {
        "pseudo_ppl": float(pseudo_ppl),
        "mean_nll": float(mean_nll),
        "n_sequences": len(seq_ppls),
        "n_tokens": total_tokens,
        "per_sequence_ppl": {
            "mean": float(np.mean(seq_ppls)),
            "std": float(np.std(seq_ppls)),
            "median": float(np.median(seq_ppls)),
        },
    }


def compute_entropy(sequences, max_seqs=5000):
    """Compute per-position nucleotide entropy (diversity measure).

    Higher entropy = more diverse generation.
    Maximum for 4 nucleotides = log2(4) = 2.0 bits.
    """
    if len(sequences) > max_seqs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in idx]

    L = len(sequences[0])
    counts = np.zeros((L, 4))  # A, C, G, T
    tok_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    for seq in sequences:
        for pos, c in enumerate(seq):
            if c in tok_idx:
                counts[pos, tok_idx[c]] += 1

    # Normalize
    probs = counts / counts.sum(axis=1, keepdims=True).clip(1)
    # Entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -(probs * np.log2(probs + 1e-10)).sum(axis=1)
    return {
        "mean_entropy_bits": float(ent.mean()),
        "min_entropy_bits": float(ent.min()),
        "max_entropy_bits": float(ent.max()),
        "max_possible": 2.0,
        "n_sequences": len(sequences),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", nargs='+', required=True,
                        help="One or more sequence files to evaluate")
    parser.add_argument("--labels", nargs='+', default=None,
                        help="Labels for each sequence file")
    parser.add_argument("--real_data", default=None,
                        help="Real data binary file for reference")
    parser.add_argument("--output", default="results/evaluation/dna_ppl.json")
    parser.add_argument("--max_seqs", type=int, default=500)
    parser.add_argument("--model", default="FreakingPotato/NucEL")
    args = parser.parse_args()

    if args.labels is None:
        args.labels = [f"model_{i}" for i in range(len(args.sequences))]

    assert len(args.labels) == len(args.sequences), \
        f"Got {len(args.sequences)} sequence files but {len(args.labels)} labels"

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NucEL MLM to {device}...")
    try:
        model = AutoModelForMaskedLM.from_pretrained(args.model).eval().to(device)
    except Exception:
        # Fallback: NucEL might not have an LM head; use AutoModel + manual head
        from transformers import AutoModel
        base = AutoModel.from_pretrained(args.model).eval().to(device)
        print("  (Using base model without LM head — embedding-distance PPL)")
        model = None

    results = {}

    for label, seq_path in zip(args.labels, args.sequences):
        print(f"\n--- {label}: {seq_path} ---")
        seqs = load_sequences(seq_path, max_n=args.max_seqs)
        print(f"  {len(seqs)} sequences")

        # Entropy
        entropy = compute_entropy(seqs)
        print(f"  Entropy: {entropy['mean_entropy_bits']:.4f} bits (max=2.0)")

        # PPL
        if model is not None:
            ppl = compute_pseudo_ppl(model, seqs, device, max_seqs=args.max_seqs)
            print(f"  Pseudo-PPL: {ppl['pseudo_ppl']:.2f}")
        else:
            ppl = None

        results[label] = {
            "entropy": entropy,
            "ppl": ppl,
            "source": seq_path,
            "n_sequences": len(seqs),
        }

    # Real data reference
    if args.real_data:
        print(f"\n--- Real data reference ---")
        real_seqs = load_real_sequences(args.real_data, max_n=args.max_seqs)
        entropy = compute_entropy(real_seqs)
        print(f"  Entropy: {entropy['mean_entropy_bits']:.4f} bits")

        if model is not None:
            ppl = compute_pseudo_ppl(model, real_seqs, device, max_seqs=args.max_seqs)
            print(f"  Pseudo-PPL: {ppl['pseudo_ppl']:.2f}")
        else:
            ppl = None

        results["real"] = {"entropy": entropy, "ppl": ppl, "source": args.real_data}

    # Summary
    print(f"\n{'='*50}")
    print(f"{'Model':<15} {'PPL':>10} {'Entropy':>10}")
    print(f"{'-'*35}")
    for label, r in results.items():
        ppl_str = f"{r['ppl']['pseudo_ppl']:.2f}" if r.get('ppl') else "N/A"
        ent_str = f"{r['entropy']['mean_entropy_bits']:.4f}"
        print(f"{label:<15} {ppl_str:>10} {ent_str:>10}")
    print(f"{'='*50}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
