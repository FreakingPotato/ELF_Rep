#!/usr/bin/env python3
"""
Evaluate ELF-B generated DNA sequences against real data.
Run AFTER sampling completes.

Usage:
  python src/evaluate_generated.py \
    --generated results/generated/generated_sequences.txt \
    --real_data /home/stark/.cache/dna-diffusion/nucel_data/train_1024.bin \
    --output results/evaluation/
"""
import argparse, json, os, sys, time
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, wasserstein_distance, ks_2samp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluate.metrics import (
    compute_gc_content, compare_gc_distribution, compare_kmer_distribution,
    compare_composition, compare_motifs, compute_diversity, compute_novelty,
    compute_sfid,
)


def load_real_sequences(bin_path, seq_len=1024, max_samples=None, seed=42):
    """Load real DNA from uint16 binary. Tokens: 11=A,12=C,13=G,14=T"""
    tmap = {11:'A', 12:'C', 13:'G', 14:'T'}
    data = np.fromfile(bin_path, dtype=np.uint16)
    n = data.shape[0] // seq_len
    data = data[:n*seq_len].reshape(n, seq_len)
    good = np.mean((data >= 11) & (data <= 14), axis=1) == 1.0
    data = data[good]
    if max_samples and len(data) > max_samples:
        idx = np.random.default_rng(seed).choice(len(data), max_samples, replace=False)
        data = data[idx]
    return [''.join(tmap.get(t,'N') for t in row) for row in data]


def generate_random_baseline(n, seq_len, gc=0.5, seed=42):
    rng = np.random.default_rng(seed)
    p = [gc/2, gc/2, (1-gc)/2, (1-gc)/2]
    return [''.join(rng.choice(['A','C','G','T'], size=seq_len, p=p)) for _ in range(n)]


def print_table(results, random_results):
    print(f"\n{'='*64}")
    print(f"{'EVALUATION SUMMARY':^64}")
    print(f"{'='*64}")
    print(f"{'Metric':<35} {'ELF-B':>10} {'Random':>10}")
    print(f"{'-'*55}")

    gc = results["gc_content"]; gc_r = random_results["gc_content"]
    print(f"{'GC Wasserstein distance':<35} {gc['wasserstein_distance']:>10.4f} {gc_r['wasserstein_distance']:>10.4f}")
    print(f"{'GC KS statistic':<35} {gc['ks_statistic']:>10.4f} {gc_r['ks_statistic']:>10.4f}")

    for k in [3,4,5,6]:
        km = results["kmer"][f"k={k}"]; km_r = random_results["kmer"][f"k={k}"]
        print(f"{f'{k}-mer cosine similarity':<35} {km['cosine_similarity']:>10.4f} {km_r['cosine_similarity']:>10.4f}")

    comp = results["composition"]
    print(f"{'Nucleotide L1 distance':<35} {comp['nucleotide_l1']:>10.4f}")
    print(f"{'Dinucleotide cosine':<35} {comp['dinucleotide_cosine']:>10.4f}")

    div_r = results["diversity_real"]; div_g = results["diversity_gen"]
    print(f"{'Diversity (real) Hamming':<35} {div_r['mean_hamming']:>10.4f}")
    print(f"{'Diversity (gen) Hamming':<35} {div_g['mean_hamming']:>10.4f}")
    print(f"{'Unique ratio (gen)':<35} {div_g['unique_ratio']:>10.4f}")

    nov = results["novelty"]
    print(f"{'Novelty (20-mer sharing rate)':<35} {nov['mean_sharing_rate']:>10.4f}")

    motif = results["motifs"]
    print(f"{'Motif freq Pearson r':<35} {motif['frequency_correlation']['pearson_r']:>10.4f}")

    sfid = results.get("s_fid")
    if sfid is not None:
        print(f"{'S-FID':<35} {sfid:>10.2f}")
    print(f"{'='*64}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--real_data", required=True)
    ap.add_argument("--output", default="results/evaluation/")
    ap.add_argument("--n_real", type=int, default=10000)
    ap.add_argument("--skip_sfid", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load generated
    print("Loading generated sequences...")
    with open(args.generated) as f:
        gen_seqs = [l.strip() for l in f if l.strip()]
    print(f"  Generated: {len(gen_seqs)} sequences")

    # Load real
    print("Loading real sequences...")
    real_seqs = load_real_sequences(args.real_data, max_samples=args.n_real)
    print(f"  Real: {len(real_seqs)} sequences")

    # Random baseline (matching real GC)
    real_gc = np.mean(compute_gc_content(real_seqs))
    random_seqs = generate_random_baseline(len(gen_seqs), len(gen_seqs[0]), gc=real_gc)
    print(f"  Random baseline: {len(random_seqs)} (GC={real_gc:.3f})")

    # Metrics
    print("\nComputing metrics...")
    from evaluate.metrics import run_full_evaluation

    real_emb, gen_emb = None, None
    if not args.skip_sfid:
        try:
            import torch
            from transformers import AutoModel
            from tqdm import tqdm
            print("Computing NucEL embeddings for S-FID...")
            nucel = AutoModel.from_pretrained("FreakingPotato/NucEL").eval()
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            nucel = nucel.to(dev)
            tmap = {'A':11,'C':12,'G':13,'T':14}

            def encode(seqs, bs=64, max_n=2000):
                rng = np.random.default_rng(42)
                if len(seqs) > max_n:
                    seqs = [seqs[i] for i in rng.choice(len(seqs), max_n, replace=False)]
                embs = []
                for i in tqdm(range(0, len(seqs), bs), desc="Encoding"):
                    batch = seqs[i:i+bs]
                    ids = torch.tensor([[tmap.get(c,0) for c in s] for s in batch], dtype=torch.long).to(dev)
                    with torch.no_grad():
                        embs.append(nucel(ids).last_hidden_state.mean(dim=1).cpu().numpy())
                return np.vstack(embs)

            real_emb = encode(real_seqs)
            gen_emb = encode(gen_seqs)
        except Exception as e:
            print(f"  Skipping S-FID: {e}")

    results = run_full_evaluation(real_seqs, gen_seqs, real_emb, gen_emb)

    # Random baseline metrics
    print("Computing random baseline...")
    from evaluate.metrics import compare_gc_distribution, compare_kmer_distribution
    rand_res = {"gc_content": compare_gc_distribution(real_seqs, random_seqs),
                "kmer": compare_kmer_distribution(real_seqs, random_seqs)}

    # Save
    all_res = {"elf_b": results, "random_baseline": rand_res,
               "n_real": len(real_seqs), "n_generated": len(gen_seqs)}
    with open(os.path.join(args.output, "metrics.json"), 'w') as f:
        json.dump(all_res, f, indent=2, default=str)

    # Print summary
    print_table(results, rand_res)

    # Plots
    print("\nGenerating plots...")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluate.visualize import generate_all_plots
    generate_all_plots(real_seqs, gen_seqs,
                       os.path.join(args.output, "figures"),
                       real_embeddings=real_emb, gen_embeddings=gen_emb)

    print(f"\nAll results saved to {args.output}/")


if __name__ == "__main__":
    main()
