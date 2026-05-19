#!/usr/bin/env python3
"""
Visualization module for DNA generation evaluation.

Produces publication-quality figures:
1. t-SNE/UMAP embedding visualization
2. Motif positional frequency plots
3. GC content distribution histograms
4. K-mer frequency comparison
5. Denoising steps curve
"""
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Dark theme
plt.style.use('dark_background')
COLORS = {'real': '#4FC3F7', 'gen': '#FF8A65', 'random': '#AED581'}
FIGSIZE = (10, 6)
DPI = 150


def plot_embeddings_tsne(real_embeddings, gen_embeddings, output_path,
                         n_samples=5000, title="t-SNE: Real vs Generated DNA Embeddings"):
    """Plot t-SNE visualization of real vs generated embeddings.

    Args:
        real_embeddings: [N, hidden] pooled embeddings
        gen_embeddings: [M, hidden] pooled embeddings
    """
    from sklearn.manifold import TSNE

    # Subsample if too large
    rng = np.random.default_rng(42)
    if len(real_embeddings) > n_samples:
        idx = rng.choice(len(real_embeddings), n_samples, replace=False)
        real_emb = real_embeddings[idx]
    else:
        real_emb = real_embeddings

    if len(gen_embeddings) > n_samples:
        idx = rng.choice(len(gen_embeddings), n_samples, replace=False)
        gen_emb = gen_embeddings[idx]
    else:
        gen_emb = gen_embeddings

    # Pool if 3D
    if real_emb.ndim == 3:
        real_emb = real_emb.mean(axis=1)
    if gen_emb.ndim == 3:
        gen_emb = gen_emb.mean(axis=1)

    # Combine
    all_emb = np.vstack([real_emb, gen_emb])
    labels = np.array(['Real'] * len(real_emb) + ['Generated'] * len(gen_emb))

    print(f"Running t-SNE on {len(all_emb)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    emb_2d = tsne.fit_transform(all_emb)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=DPI)
    mask_real = labels == 'Real'
    ax.scatter(emb_2d[mask_real, 0], emb_2d[mask_real, 1],
               c=COLORS['real'], alpha=0.3, s=2, label='Real', rasterized=True)
    ax.scatter(emb_2d[~mask_real, 0], emb_2d[~mask_real, 1],
               c=COLORS['gen'], alpha=0.3, s=2, label='Generated', rasterized=True)
    ax.legend(fontsize=12, markerscale=5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('t-SNE 1', fontsize=12)
    ax.set_ylabel('t-SNE 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {output_path}")


def plot_gc_distribution(real_seqs, gen_seqs, output_path,
                         title="GC Content Distribution"):
    """Plot GC content histogram overlay."""
    from evaluate.metrics import compute_gc_content

    real_gc = compute_gc_content(real_seqs)
    gen_gc = compute_gc_content(gen_seqs)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, dpi=DPI)
    bins = np.linspace(0, 1, 50)
    ax.hist(real_gc, bins=bins, alpha=0.6, color=COLORS['real'], label='Real', density=True)
    ax.hist(gen_gc, bins=bins, alpha=0.6, color=COLORS['gen'], label='Generated', density=True)
    ax.set_xlabel('GC Content', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)

    # Add stats text
    from scipy.stats import ks_2samp, wasserstein_distance
    wd = wasserstein_distance(real_gc, gen_gc)
    ks_stat, ks_pval = ks_2samp(real_gc, gen_gc)
    stats_text = f'Wasserstein: {wd:.4f}\nKS stat: {ks_stat:.4f}\nKS p-val: {ks_pval:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved GC plot to {output_path}")


def plot_kmer_comparison(real_seqs, gen_seqs, output_path, k=4,
                         top_n=20, title=None):
    """Plot k-mer frequency comparison bar chart."""
    from evaluate.metrics import compute_kmer_frequencies

    real_freq = compute_kmer_frequencies(real_seqs, k)
    gen_freq = compute_kmer_frequencies(gen_seqs, k)

    # Sort by real frequency, take top N
    all_kmers = sorted(real_freq.keys(), key=lambda x: real_freq[x], reverse=True)[:top_n]

    real_vals = [real_freq.get(km, 0) for km in all_kmers]
    gen_vals = [gen_freq.get(km, 0) for km in all_kmers]

    if title is None:
        title = f'{k}-mer Frequency: Top {top_n}'

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=DPI)
    x = np.arange(len(all_kmers))
    width = 0.35
    ax.bar(x - width/2, real_vals, width, color=COLORS['real'], label='Real', alpha=0.8)
    ax.bar(x + width/2, gen_vals, width, color=COLORS['gen'], label='Generated', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_kmers, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel(f'{k}-mer', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved k-mer plot to {output_path}")


def plot_motif_positional(real_seqs, gen_seqs, motifs, output_path,
                          title="Motif Positional Frequency"):
    """Plot positional motif frequency (like DiscDiff Figure 1).

    Args:
        motifs: dict of name -> motif_string
    """
    from evaluate.metrics import compute_motif_positional_frequency

    n_motifs = len(motifs)
    fig, axes = plt.subplots(1, n_motifs, figsize=(6 * n_motifs, 4), dpi=DPI)
    if n_motifs == 1:
        axes = [axes]

    for ax, (name, motif) in zip(axes, motifs.items()):
        real_pos = compute_motif_positional_frequency(real_seqs, motif)
        gen_pos = compute_motif_positional_frequency(gen_seqs, motif)

        # Smooth with moving average
        window = 10
        if len(real_pos) > window:
            kernel = np.ones(window) / window
            real_smooth = np.convolve(real_pos, kernel, mode='same')
            gen_smooth = np.convolve(gen_pos, kernel, mode='same')
        else:
            real_smooth = real_pos
            gen_smooth = gen_pos

        ax.plot(real_smooth, color=COLORS['real'], label='Real', alpha=0.8)
        ax.plot(gen_smooth, color=COLORS['gen'], label='Generated', alpha=0.8)
        ax.set_title(f'{name} ({motif})', fontsize=11)
        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(fontsize=9)

        if real_pos.std() > 0 and gen_pos.std() > 0:
            r, p = pearsonr(real_pos, gen_pos)
            ax.text(0.02, 0.98, f'r={r:.3f}', transform=ax.transAxes, fontsize=9,
                    verticalalignment='top')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved motif positional plot to {output_path}")


def plot_dinucleotide_comparison(real_seqs, gen_seqs, output_path,
                                 title="Dinucleotide Frequency"):
    """Plot dinucleotide frequency comparison."""
    from evaluate.metrics import compute_dinucleotide_composition

    real_di = compute_dinucleotide_composition(real_seqs)
    gen_di = compute_dinucleotide_composition(gen_seqs)

    all_di = sorted(set(list(real_di.keys()) + list(gen_di.keys())))
    real_vals = [real_di.get(d, 0) for d in all_di]
    gen_vals = [gen_di.get(d, 0) for d in all_di]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6), dpi=DPI)
    x = np.arange(len(all_di))
    width = 0.35
    ax.bar(x - width/2, real_vals, width, color=COLORS['real'], label='Real', alpha=0.8)
    ax.bar(x + width/2, gen_vals, width, color=COLORS['gen'], label='Generated', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_di, rotation=45, ha='right', fontsize=10)
    ax.set_xlabel('Dinucleotide', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved dinucleotide plot to {output_path}")


def generate_all_plots(real_seqs, gen_seqs, output_dir,
                       real_embeddings=None, gen_embeddings=None):
    """Generate all evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. GC Content
    plot_gc_distribution(real_seqs, gen_seqs,
                         os.path.join(output_dir, "gc_content.png"))

    # 2. K-mer comparison (k=3,4,5)
    for k in [3, 4, 5]:
        plot_kmer_comparison(real_seqs, gen_seqs,
                             os.path.join(output_dir, f"kmer_k{k}.png"), k=k)

    # 3. Dinucleotide
    plot_dinucleotide_comparison(real_seqs, gen_seqs,
                                 os.path.join(output_dir, "dinucleotide.png"))

    # 4. Motif positional
    motifs_to_plot = {
        "TATA-box": "TATAAA",
        "CCAAT": "CCAAT",
        "GC-box": "GGGCGG",
    }
    plot_motif_positional(real_seqs, gen_seqs, motifs_to_plot,
                          os.path.join(output_dir, "motif_positional.png"))

    # 5. t-SNE (if embeddings provided)
    if real_embeddings is not None and gen_embeddings is not None:
        plot_embeddings_tsne(real_embeddings, gen_embeddings,
                             os.path.join(output_dir, "tsne.png"))

    print(f"\nAll plots saved to {output_dir}/")
