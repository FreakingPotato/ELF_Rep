#!/usr/bin/env python3
"""
T6+S7: Compute NucEL embeddings → S-FID + t-SNE visualization.
"""
import sys, os, json, argparse, time
import numpy as np
import torch
from transformers import AutoModel
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evaluate.metrics import compute_sfid

TOKEN_MAP = {'A': 11, 'C': 12, 'G': 13, 'T': 14}

def load_seqs(path, max_n=None):
    with open(path) as f:
        seqs = [l.strip() for l in f if l.strip()]
    if max_n and len(seqs) > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(seqs), max_n, replace=False)
        seqs = [seqs[i] for i in idx]
    return seqs

def load_real_seqs(bin_path, seq_len=1024, max_n=None):
    tmap = {11:'A', 12:'C', 13:'G', 14:'T'}
    data = np.fromfile(bin_path, dtype=np.uint16)
    n = data.shape[0] // seq_len
    data = data[:n*seq_len].reshape(n, seq_len)
    good = np.mean((data >= 11) & (data <= 14), axis=1) == 1.0
    data = data[good]
    if max_n and len(data) > max_n:
        idx = np.random.default_rng(42).choice(len(data), max_n, replace=False)
        data = data[idx]
    return [''.join(tmap.get(t,'N') for t in row) for row in data]

def encode_seqs(model, seqs, device, batch_size=64, pool='mean'):
    """Encode DNA sequences → NucEL embeddings [N, 512]."""
    embs = []
    for i in tqdm(range(0, len(seqs), batch_size), desc=f"Encoding ({pool})"):
        batch = seqs[i:i+batch_size]
        ids = torch.tensor(
            [[TOKEN_MAP.get(c, 0) for c in s] for s in batch],
            dtype=torch.long, device=device
        )
        with torch.no_grad():
            out = model(ids).last_hidden_state  # [B, L, 512]
            if pool == 'mean':
                emb = out.mean(dim=1)
            elif pool == 'first':
                emb = out[:, 0]
            else:
                emb = out.reshape(out.shape[0], -1)  # flatten
        embs.append(emb.cpu().numpy())
    return np.vstack(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default="results/generated/generated_sequences.txt")
    ap.add_argument("--real_data", default="/home/stark/.cache/dna-diffusion/nucel_data/train_1024.bin")
    ap.add_argument("--output", default="results/evaluation/")
    ap.add_argument("--n_encode", type=int, default=2000, help="Sequences to encode for S-FID")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "figures"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NucEL encoder to {device}...")
    model = AutoModel.from_pretrained("FreakingPotato/NucEL").eval().to(device)

    print("Loading sequences...")
    gen_seqs = load_seqs(args.generated, max_n=args.n_encode)
    real_seqs = load_real_seqs(args.real_data, max_n=args.n_encode)
    print(f"  Gen: {len(gen_seqs)}, Real: {len(real_seqs)}")

    # Encode
    print("\nEncoding generated sequences...")
    gen_emb = encode_seqs(model, gen_seqs, device)
    print("Encoding real sequences...")
    real_emb = encode_seqs(model, real_seqs, device)
    print(f"  Gen embeddings: {gen_emb.shape}, Real: {real_emb.shape}")

    # Save embeddings
    np.save(os.path.join(args.output, "real_nucel_emb.npy"), real_emb)
    np.save(os.path.join(args.output, "gen_nucel_emb.npy"), gen_emb)

    # T6: S-FID
    print("\n--- T6: Computing S-FID ---")
    sfid = compute_sfid(real_emb, gen_emb)
    print(f"S-FID = {sfid:.2f}")

    # Update metrics.json
    metrics_path = os.path.join(args.output, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics.setdefault("elf_b", {})["s_fid"] = sfid
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Updated {metrics_path}")

    # T7: t-SNE
    print("\n--- T7: Computing t-SNE ---")
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    X = np.vstack([real_emb, gen_emb])
    labels = np.array(['Real'] * len(real_emb) + ['Generated'] * len(gen_emb))
    print(f"Running t-SNE on {X.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    for label, color in [('Real', '#4C72B0'), ('Generated', '#DD8452')]:
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=label, alpha=0.5, s=10)
    ax.legend(fontsize=12)
    ax.set_title('t-SNE: Real vs Generated (NucEL Embeddings)', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    fig.tight_layout()
    fig_path = os.path.join(args.output, "figures", "tsne.png")
    fig.savefig(fig_path, dpi=150)
    print(f"Saved t-SNE plot to {fig_path}")

    np.save(os.path.join(args.output, "tsne_coords.npy"), X_2d)
    np.save(os.path.join(args.output, "tsne_labels.npy"), labels)

    print(f"\n✅ T6 S-FID = {sfid:.2f}")
    print(f"✅ T7 t-SNE saved")
    print("Done!")


if __name__ == "__main__":
    main()
