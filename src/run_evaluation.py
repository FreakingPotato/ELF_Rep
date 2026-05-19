#!/usr/bin/env python3
"""
Run full evaluation pipeline for ELF-B + NucEL DNA generation model.

1. Generate sequences (or load existing)
2. Load real sequences for comparison
3. Compute all metrics
4. Generate all plots

Usage:
  python src/run_evaluation.py \
    --checkpoint outputs/elf-b-hg38-nucel/checkpoint_48870 \
    --config configs/training_configs/train_hg38_nucel_ELF-B.yml \
    --real_data /home/stark/.cache/dna-diffusion/nucel_data/train_1024.bin \
    --output results/evaluation/
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_real_sequences(bin_path, seq_len=1024, max_samples=None):
    """Load real DNA sequences from uint16 binary file.

    Token mapping: 11=A, 12=C, 13=G, 14=T
    """
    token_map = {11: 'A', 12: 'C', 13: 'G', 14: 'T'}
    data = np.fromfile(bin_path, dtype=np.uint16)
    n_samples = data.shape[0] // seq_len
    data = data[:n_samples * seq_len].reshape(n_samples, seq_len)

    # Filter to pure nucleotide sequences
    is_nuc = (data >= 11) & (data <= 14)
    nuc_frac = np.mean(is_nuc, axis=1)
    good_mask = nuc_frac == 1.0
    data = data[good_mask]

    if max_samples and len(data) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), max_samples, replace=False)
        data = data[idx]

    sequences = []
    for row in data:
        seq = ''.join(token_map.get(t, 'N') for t in row)
        sequences.append(seq)
    return sequences


def generate_random_baseline(n_samples, seq_len, gc_content=0.5, seed=42):
    """Generate random DNA sequences matching GC content."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(n_samples):
        p = [gc_content/2, gc_content/2, (1-gc_content)/2, (1-gc_content)/2]
        tokens = rng.choice(['A', 'C', 'G', 'T'], size=seq_len, p=p)
        sequences.append(''.join(tokens))
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/elf-b-hg38-nucel/checkpoint_48870")
    parser.add_argument("--config", default="configs/training_configs/train_hg38_nucel_ELF-B.yml")
    parser.add_argument("--real_data", default="/home/stark/.cache/dna-diffusion/nucel_data/train_1024.bin")
    parser.add_argument("--generated", default=None, help="Pre-generated sequences file (skip sampling)")
    parser.add_argument("--output", default="results/evaluation/")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_real", type=int, default=10000, help="Number of real sequences to compare")
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--skip_sampling", action="store_true")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip S-FID (needs GPU)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print(f"{'='*60}")
    print(f"ELF-B + NucEL Evaluation Pipeline")
    print(f"{'='*60}")

    # ---- Step 1: Load or generate sequences ----
    gen_path = os.path.join(args.output, "generated_sequences.txt")

    if args.generated:
        # Load pre-generated
        print(f"Loading pre-generated sequences from {args.generated}")
        with open(args.generated) as f:
            gen_seqs = [line.strip() for line in f if line.strip()]
    elif os.path.exists(gen_path) and not args.skip_sampling:
        print(f"Loading existing generated sequences from {gen_path}")
        with open(gen_path) as f:
            gen_seqs = [line.strip() for line in f if line.strip()]
    else:
        print("\n--- Phase 1: Sampling ---")
        from sample_nucel import main as sample_main
        # Run sampling via exec
        import subprocess
        cmd = [
            sys.executable, "src/sample_nucel.py",
            "--checkpoint", args.checkpoint,
            "--config", args.config,
            "--n_samples", str(args.n_samples),
            "--sampling_steps", str(args.sampling_steps),
            "--batch_size", str(args.batch_size),
            "--output", args.output,
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        with open(gen_path) as f:
            gen_seqs = [line.strip() for line in f if line.strip()]

    print(f"Generated sequences: {len(gen_seqs)}")

    # ---- Step 2: Load real sequences ----
    print(f"\n--- Phase 2: Loading real sequences ---")
    real_seqs = load_real_sequences(args.real_data, max_samples=args.n_real)
    print(f"Real sequences: {len(real_seqs)}")

    # ---- Step 3: Generate baselines ----
    print(f"\n--- Phase 3: Generating baselines ---")
    # Match GC content of real data
    from evaluate.metrics import compute_gc_content
    real_gc = np.mean(compute_gc_content(real_seqs))
    random_seqs = generate_random_baseline(len(gen_seqs), len(gen_seqs[0]), gc_content=real_gc)
    print(f"Random baseline: {len(random_seqs)} sequences (GC={real_gc:.3f})")

    # ---- Step 4: Compute metrics ----
    print(f"\n--- Phase 4: Computing metrics ---")
    from evaluate.metrics import run_full_evaluation

    # Compute embeddings for S-FID if possible
    real_emb = None
    gen_emb = None
    if not args.skip_embeddings:
        try:
            print("Computing NucEL embeddings for S-FID...")
            import torch
            from transformers import AutoModel
            from tqdm import tqdm

            model = AutoModel.from_pretrained("FreakingPotato/NucEL")
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            def encode_seqs(seqs, batch_size=32, max_samples=2000):
                token_map = {'A': 11, 'C': 12, 'G': 13, 'T': 14}
                all_emb = []
                rng = np.random.default_rng(42)
                if len(seqs) > max_samples:
                    idx = rng.choice(len(seqs), max_samples, replace=False)
                    seqs_sub = [seqs[i] for i in idx]
                else:
                    seqs_sub = seqs

                for i in tqdm(range(0, len(seqs_sub), batch_size), desc="Encoding"):
                    batch = seqs_sub[i:i+batch_size]
                    tokens = []
                    for s in batch:
                        t = [token_map.get(c, 0) for c in s]
                        tokens.append(t)
                    input_ids = torch.tensor(tokens, dtype=torch.long).to(device)
                    with torch.no_grad():
                        out = model(input_ids)
                        # Use last hidden state, pool over sequence
                        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
                    all_emb.append(emb)
                return np.vstack(all_emb)

            real_emb = encode_seqs(real_seqs)
            gen_emb = encode_seqs(gen_seqs)
            print(f"Real embeddings: {real_emb.shape}, Gen embeddings: {gen_emb.shape}")
        except Exception as e:
            print(f"Skipping embeddings (error): {e}")

    results = run_full_evaluation(real_seqs, gen_seqs, real_emb, gen_emb)

    # Also evaluate random baseline
    print("\nComputing random baseline metrics...")
    random_results = {}
    from evaluate.metrics import compare_gc_distribution, compare_kmer_distribution, compare_composition
    random_results["gc_content"] = compare_gc_distribution(real_seqs, random_seqs)
    random_results["kmer"] = compare_kmer_distribution(real_seqs, random_seqs)

    # ---- Step 5: Save metrics ----
    print(f"\n--- Phase 5: Saving results ---")
    all_results = {
        "elf_b": results,
        "random_baseline": random_results,
        "n_real": len(real_seqs),
        "n_generated": len(gen_seqs),
    }

    results_path = os.path.join(args.output, "metrics.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved metrics to {results_path}")

    # ---- Step 6: Generate plots ----
    print(f"\n--- Phase 6: Generating plots ---")
    from evaluate.visualize import generate_all_plots
    generate_all_plots(
        real_seqs, gen_seqs,
        os.path.join(args.output, "figures"),
        real_embeddings=real_emb,
        gen_embeddings=gen_emb,
    )

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'ELF-B':>12} {'Random':>12}")
    print(f"{'-'*54}")

    gc = results["gc_content"]
    gc_r = random_results["gc_content"]
    print(f"{'GC Wasserstein dist':<30} {gc['wasserstein_distance']:>12.4f} {gc_r['wasserstein_distance']:>12.4f}")
    print(f"{'GC KS statistic':<30} {gc['ks_statistic']:>12.4f} {gc_r['ks_statistic']:>12.4f}")

    for k in [3, 4, 5]:
        km = results["kmer"][f"k={k}"]
        km_r = random_results["kmer"][f"k={k}"]
        print(f"{f'{k}-mer cosine sim':<30} {km['cosine_similarity']:>12.4f} {km_r['cosine_similarity']:>12.4f}")

    comp = results["composition"]
    print(f"{'Nucleotide L1 dist':<30} {comp['nucleotide_l1']:>12.4f}")
    print(f"{'Dinucleotide cosine':<30} {comp['dinucleotide_cosine']:>12.4f}")

    div_r = results["diversity_real"]
    div_g = results["diversity_gen"]
    print(f"{'Diversity (real) Hamming':<30} {div_r['mean_hamming']:>12.4f}")
    print(f"{'Diversity (gen) Hamming':<30} {div_g['mean_hamming']:>12.4f}")
    print(f"{'Unique ratio (real)':<30} {div_r['unique_ratio']:>12.4f}")
    print(f"{'Unique ratio (gen)':<30} {div_g['unique_ratio']:>12.4f}")

    nov = results["novelty"]
    print(f"{'Novelty (20-mer sharing)':<30} {nov['mean_sharing_rate']:>12.4f}")

    motif = results["motifs"]
    print(f"{'Motif freq correlation':<30} {motif['frequency_correlation']['pearson_r']:>12.4f}")

    if results.get("s_fid") is not None:
        print(f"{'S-FID':<30} {results['s_fid']:>12.2f}")

    print(f"{'='*60}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
