#!/usr/bin/env python3
"""
Evaluation metrics for DNA sequence generation models.

Implements metrics from DNA-Diffusion, DiscDiff, DDSM:
- S-FID (Sequence Fréchet Inception Distance)
- Motif frequency correlation
- GC content distribution comparison
- K-mer distribution comparison
- Nucleotide composition comparison
- Sequence diversity
- Sequence novelty
"""
import json
import os
import sys
import numpy as np
from collections import Counter
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, wasserstein_distance, ks_2samp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================
# S-FID (Sequence Fréchet Inception Distance)
# ============================================

def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Fréchet Inception Distance between two Gaussian distributions."""
    from scipy.linalg import sqrtm
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_sfid(real_embeddings, gen_embeddings):
    """Compute S-FID using NucEL embeddings.

    Args:
        real_embeddings: [N_real, seq_len, hidden] or [N_real, hidden] (pooled)
        gen_embeddings: [N_gen, seq_len, hidden] or [N_gen, hidden] (pooled)

    Returns:
        S-FID score (lower is better)
    """
    # Pool over sequence dimension if needed
    if real_embeddings.ndim == 3:
        real_embeddings = real_embeddings.mean(axis=1)
    if gen_embeddings.ndim == 3:
        gen_embeddings = gen_embeddings.mean(axis=1)

    mu_real = real_embeddings.mean(axis=0)
    mu_gen = gen_embeddings.mean(axis=0)
    sigma_real = np.cov(real_embeddings, rowvar=False)
    sigma_gen = np.cov(gen_embeddings, rowvar=False)

    return compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)


# ============================================
# GC Content Distribution
# ============================================

def compute_gc_content(sequences):
    """Compute GC content for each sequence.

    Returns: array of GC fractions
    """
    gc_contents = []
    for seq in sequences:
        if len(seq) == 0:
            continue
        gc = (seq.upper().count('G') + seq.upper().count('C')) / len(seq)
        gc_contents.append(gc)
    return np.array(gc_contents)


def compare_gc_distribution(real_seqs, gen_seqs):
    """Compare GC content distributions.

    Returns dict with:
        wasserstein: Wasserstein distance (lower = more similar)
        ks_stat: Kolmogorov-Smirnov statistic
        ks_pvalue: KS test p-value
        real_mean, gen_mean: mean GC contents
    """
    real_gc = compute_gc_content(real_seqs)
    gen_gc = compute_gc_content(gen_seqs)

    wd = wasserstein_distance(real_gc, gen_gc)
    ks_stat, ks_pval = ks_2samp(real_gc, gen_gc)

    return {
        "wasserstein_distance": float(wd),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "real_mean": float(real_gc.mean()),
        "real_std": float(real_gc.std()),
        "gen_mean": float(gen_gc.mean()),
        "gen_std": float(gen_gc.std()),
    }


# ============================================
# K-mer Distribution
# ============================================

def compute_kmer_frequencies(sequences, k=4):
    """Compute k-mer frequency distribution across all sequences.

    Returns: dict of kmer -> frequency
    """
    kmer_counts = Counter()
    total = 0
    for seq in sequences:
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(c in 'ACGT' for c in kmer):
                kmer_counts[kmer] += 1
                total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in kmer_counts.items()}


def compare_kmer_distribution(real_seqs, gen_seqs, k_values=[3, 4, 5, 6]):
    """Compare k-mer distributions between real and generated sequences.

    Returns dict with cosine similarity and Pearson correlation for each k.
    """
    results = {}
    for k in k_values:
        real_freq = compute_kmer_frequencies(real_seqs, k)
        gen_freq = compute_kmer_frequencies(gen_seqs, k)

        # Get all kmers
        all_kmers = sorted(set(list(real_freq.keys()) + list(gen_freq.keys())))
        real_vec = np.array([real_freq.get(km, 0) for km in all_kmers])
        gen_vec = np.array([gen_freq.get(km, 0) for km in all_kmers])

        # Cosine similarity
        if np.linalg.norm(real_vec) > 0 and np.linalg.norm(gen_vec) > 0:
            cos_sim = 1 - cosine(real_vec, gen_vec)
        else:
            cos_sim = 0.0

        # Pearson correlation
        if len(real_vec) > 2:
            pearson_r, pearson_p = pearsonr(real_vec, gen_vec)
        else:
            pearson_r, pearson_p = 0.0, 1.0

        # L1 distance (total variation)
        l1_dist = np.abs(real_vec - gen_vec).sum() / 2

        results[f"k={k}"] = {
            "cosine_similarity": float(cos_sim),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "l1_distance": float(l1_dist),
            "n_unique_kmers": len(all_kmers),
        }
    return results


# ============================================
# Nucleotide Composition
# ============================================

def compute_nucleotide_composition(sequences):
    """Compute per-position nucleotide frequencies.

    Returns: dict of nucleotide -> frequency
    """
    counts = Counter()
    total = 0
    for seq in sequences:
        seq = seq.upper()
        for c in seq:
            if c in 'ACGT':
                counts[c] += 1
                total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def compute_dinucleotide_composition(sequences):
    """Compute dinucleotide frequency distribution."""
    counts = Counter()
    total = 0
    for seq in sequences:
        seq = seq.upper()
        for i in range(len(seq) - 1):
            di = seq[i:i+2]
            if all(c in 'ACGT' for c in di):
                counts[di] += 1
                total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def compare_composition(real_seqs, gen_seqs):
    """Compare nucleotide and dinucleotide compositions."""
    real_nuc = compute_nucleotide_composition(real_seqs)
    gen_nuc = compute_nucleotide_composition(gen_seqs)

    real_di = compute_dinucleotide_composition(real_seqs)
    gen_di = compute_dinucleotide_composition(gen_seqs)

    # Nucleotide comparison
    nuc_diff = {}
    for n in 'ACGT':
        r = real_nuc.get(n, 0)
        g = gen_nuc.get(n, 0)
        nuc_diff[n] = {"real": r, "gen": g, "diff": abs(r - g)}
    nuc_l1 = sum(d["diff"] for d in nuc_diff.values())

    # Dinucleotide comparison
    all_di = sorted(set(list(real_di.keys()) + list(gen_di.keys())))
    di_vec_r = np.array([real_di.get(d, 0) for d in all_di])
    di_vec_g = np.array([gen_di.get(d, 0) for d in all_di])
    di_cos = 1 - cosine(di_vec_r, di_vec_g) if (np.linalg.norm(di_vec_r) > 0 and np.linalg.norm(di_vec_g) > 0) else 0

    return {
        "nucleotide": nuc_diff,
        "nucleotide_l1": float(nuc_l1),
        "dinucleotide_cosine": float(di_cos),
        "dinucleotide_pearson": float(pearsonr(di_vec_r, di_vec_g)[0]) if len(di_vec_r) > 2 else 0,
    }


# ============================================
# Motif Analysis
# ============================================

# Common regulatory motifs
MOTIFS = {
    "TATA-box": "TATAAA",
    "TATA-box_alt": "TATATA",
    "Inr_CCAAT": "CCAAT",
    "GC-box": "GGGCGG",
    "Sp1": "GGGCGG",
    "CAAT-box": "CCAAT",
    "BREu": "SSRCGCC",
    "DPE": "RGWYV",
}

# Simplified motif list (exact matches)
SIMPLE_MOTIFS = {
    "TATA-box": ["TATAAA", "TATATA"],
    "GC-box": ["GGGCGG", "CCGCCC"],
    "CCAAT-box": ["CCAAT"],
    "AT-rich": ["AAAA", "TTTT"],
    "CG": ["CG"],
    "TGCA": ["TGCA"],
}


def compute_motif_frequency(sequences, motif):
    """Compute frequency of a motif across sequences."""
    count = 0
    total_positions = 0
    for seq in sequences:
        seq = seq.upper()
        positions = 0
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx == -1:
                break
            positions += 1
            start = idx + 1
        count += positions
        total_positions += len(seq) - len(motif) + 1
    return count / max(total_positions, 1)


def compute_motif_positional_frequency(sequences, motif, seq_len=None):
    """Compute positional frequency of a motif.

    Returns: array of frequency at each position
    """
    if seq_len is None:
        seq_len = max(len(s) for s in sequences)
    positional_counts = np.zeros(seq_len)
    n_seqs = len(sequences)

    for seq in sequences:
        seq = seq.upper()
        start = 0
        while True:
            idx = seq.find(motif, start)
            if idx == -1:
                break
            if idx < seq_len:
                positional_counts[idx] += 1
            start = idx + 1

    return positional_counts / n_seqs


def compare_motifs(real_seqs, gen_seqs, motifs=None):
    """Compare motif frequencies between real and generated sequences.

    Returns:
        freq_correlation: Pearson r of motif frequencies
        positional_correlations: per-motif positional Pearson r
        details: per-motif stats
    """
    if motifs is None:
        motifs = SIMPLE_MOTIFS

    real_freqs = []
    gen_freqs = []
    details = {}
    pos_correlations = {}

    for name, motif_list in motifs.items():
        for motif in motif_list:
            rf = compute_motif_frequency(real_seqs, motif)
            gf = compute_motif_frequency(gen_seqs, motif)
            real_freqs.append(rf)
            gen_freqs.append(gf)
            details[f"{name}_{motif}"] = {"real": rf, "gen": gf}

            # Positional correlation
            real_pos = compute_motif_positional_frequency(real_seqs, motif)
            gen_pos = compute_motif_positional_frequency(gen_seqs, motif)
            if real_pos.std() > 0 and gen_pos.std() > 0:
                r, p = pearsonr(real_pos, gen_pos)
            else:
                r, p = 0.0, 1.0
            pos_correlations[motif] = {"pearson_r": float(r), "pearson_p": float(p)}

    # Overall frequency correlation
    if len(real_freqs) > 2:
        overall_r, overall_p = pearsonr(real_freqs, gen_freqs)
    else:
        overall_r, overall_p = 0.0, 1.0

    return {
        "frequency_correlation": {"pearson_r": float(overall_r), "pearson_p": float(overall_p)},
        "positional_correlations": pos_correlations,
        "details": details,
    }


# ============================================
# Sequence Diversity
# ============================================

def compute_diversity(sequences, sample_size=1000):
    """Compute sequence diversity via pairwise Hamming distance.

    Uses a random subset for efficiency.
    """
    rng = np.random.default_rng(42)
    if len(sequences) > sample_size:
        indices = rng.choice(len(sequences), sample_size, replace=False)
        subset = [sequences[i] for i in indices]
    else:
        subset = sequences
        sample_size = len(subset)

    # Compute pairwise distances for random pairs
    n_pairs = min(5000, sample_size * (sample_size - 1) // 2)
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.choice(sample_size, 2, replace=False)
        pairs.append((i, j))

    distances = []
    for i, j in pairs:
        s1, s2 = subset[i], subset[j]
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            continue
        hamming = sum(c1 != c2 for c1, c2 in zip(s1[:min_len], s2[:min_len]))
        distances.append(hamming / min_len)

    return {
        "mean_hamming": float(np.mean(distances)),
        "std_hamming": float(np.std(distances)),
        "n_pairs": len(distances),
        "unique_ratio": len(set(sequences)) / len(sequences),
    }


# ============================================
# Novelty (approximate)
# ============================================

def compute_novelty(gen_seqs, real_seqs, sample_size=1000):
    """Approximate novelty via substring matching.

    For each generated sequence, find the longest common substring
    with any real sequence. Shorter = more novel.

    This is an approximation of BLAT analysis.
    """
    rng = np.random.default_rng(42)

    if len(gen_seqs) > sample_size:
        gen_indices = rng.choice(len(gen_seqs), sample_size, replace=False)
        gen_sample = [gen_seqs[i] for i in gen_indices]
    else:
        gen_sample = gen_seqs

    if len(real_seqs) > sample_size:
        real_indices = rng.choice(len(real_seqs), sample_size, replace=False)
        real_sample = [real_seqs[i] for i in real_indices]
    else:
        real_sample = real_seqs

    # For each generated seq, compute max overlap with any real seq
    # Using a sliding window approach
    max_overlaps = []
    k = 20  # check k-mer sharing

    # Build k-mer index for real sequences
    real_kmers = {}
    for seq_idx, seq in enumerate(real_sample):
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(c in 'ACGT' for c in kmer):
                if kmer not in real_kmers:
                    real_kmers[kmer] = []
                real_kmers[kmer].append(seq_idx)

    for seq in gen_sample:
        seq = seq.upper()
        shared_kmers = 0
        total_kmers = 0
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(c in 'ACGT' for c in kmer):
                total_kmers += 1
                if kmer in real_kmers:
                    shared_kmers += 1
        sharing_rate = shared_kmers / max(total_kmers, 1)
        max_overlaps.append(sharing_rate)

    return {
        "mean_sharing_rate": float(np.mean(max_overlaps)),
        "std_sharing_rate": float(np.std(max_overlaps)),
        "median_sharing_rate": float(np.median(max_overlaps)),
        "k": k,
        "note": "Fraction of k-mers in generated seq that appear in real data. Lower = more novel.",
    }


# ============================================
# Full Evaluation Pipeline
# ============================================

def run_full_evaluation(real_seqs, gen_seqs, real_embeddings=None, gen_embeddings=None):
    """Run all evaluation metrics.

    Args:
        real_seqs: list of DNA strings (real)
        gen_seqs: list of DNA strings (generated)
        real_embeddings: optional numpy array of real NucEL embeddings for S-FID
        gen_embeddings: optional numpy array of generated NucEL embeddings for S-FID

    Returns:
        dict of all metrics
    """
    results = {}

    print("Computing GC content comparison...")
    results["gc_content"] = compare_gc_distribution(real_seqs, gen_seqs)

    print("Computing k-mer distributions...")
    results["kmer"] = compare_kmer_distribution(real_seqs, gen_seqs)

    print("Computing nucleotide composition...")
    results["composition"] = compare_composition(real_seqs, gen_seqs)

    print("Computing motif comparison...")
    results["motifs"] = compare_motifs(real_seqs, gen_seqs)

    print("Computing diversity...")
    results["diversity_real"] = compute_diversity(real_seqs)
    results["diversity_gen"] = compute_diversity(gen_seqs)

    print("Computing novelty...")
    results["novelty"] = compute_novelty(gen_seqs, real_seqs)

    if real_embeddings is not None and gen_embeddings is not None:
        print("Computing S-FID...")
        results["s_fid"] = compute_sfid(real_embeddings, gen_embeddings)
    else:
        results["s_fid"] = None

    return results
