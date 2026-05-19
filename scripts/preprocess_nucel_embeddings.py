#!/usr/bin/env python3
"""
Preprocess hg38 genome data for ELF training with NucEL encoder.

Saves embeddings as sharded .npy files to avoid memory-mapped OOM issues.

Usage:
  python scripts/preprocess_nucel_embeddings.py \
    --input /path/to/train_1024.bin \
    --output data/hg38_nucel_1024/train \
    --batch_size 64 \
    --shard_size 1000
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, help="Output prefix (shards go to {output}_shards/)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--shard_size", type=int, default=500, help="Samples per shard file")
    parser.add_argument("--model_name", type=str, default="FreakingPotato/NucEL")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"NucEL Embedding Preprocessing (Sharded)")
    print(f"{'='*60}")

    # Load tokenized data (uint16)
    print(f"\nLoading tokenized data (uint16)...")
    data = np.fromfile(args.input, dtype=np.uint16)
    n_samples = data.shape[0] // args.seq_len
    data = data[:n_samples * args.seq_len].reshape(n_samples, args.seq_len)

    # Filter: keep only 100% nucleotide sequences
    is_nuc = (data >= 11) & (data <= 14)
    nuc_frac = np.mean(is_nuc, axis=1)
    good_mask = nuc_frac == 1.0
    data = data[good_mask]
    n_samples = len(data)
    print(f"  Clean sequences: {n_samples:,}")

    if args.max_samples:
        data = data[:args.max_samples]
        n_samples = len(data)
        print(f"  Limited to: {n_samples:,}")

    # Load NucEL
    print(f"\nLoading NucEL model...")
    model = AutoModel.from_pretrained(args.model_name).to(args.device).eval()
    hidden_size = model.config.hidden_size
    print(f"  Hidden size: {hidden_size}")

    # Compute stats
    print(f"\nComputing embedding statistics...")
    stat_n = min(500, n_samples)
    stat_ids = torch.tensor(data[:stat_n], dtype=torch.long).to(args.device)
    stat_embs = []
    with torch.no_grad():
        for i in range(0, len(stat_ids), args.batch_size):
            batch = stat_ids[i:i+args.batch_size]
            out = model(input_ids=batch, attention_mask=torch.ones_like(batch))
            stat_embs.append(out.last_hidden_state.cpu().float().numpy())
    stat_embs = np.concatenate(stat_embs, axis=0)
    latent_mean = float(stat_embs.mean())
    latent_std = float(stat_embs.std())
    del stat_embs, stat_ids
    torch.cuda.empty_cache()
    print(f"  Mean: {latent_mean:.6f}, Std: {latent_std:.6f}")

    # Create output dir
    shards_dir = args.output + "_shards"
    os.makedirs(shards_dir, exist_ok=True)

    # Encode in shards
    print(f"\nEncoding {n_samples:,} samples (shard_size={args.shard_size})...")
    t0 = time.time()
    n_shards = (n_samples + args.shard_size - 1) // args.shard_size

    for shard_idx in range(n_shards):
        start = shard_idx * args.shard_size
        end = min(start + args.shard_size, n_samples)
        shard_data = data[start:end]
        shard_n = len(shard_data)

        # Encode this shard
        shard_embs = np.empty((shard_n, args.seq_len, hidden_size), dtype=np.float16)
        with torch.no_grad():
            for i in range(0, shard_n, args.batch_size):
                batch_data = shard_data[i:i+args.batch_size]
                bs = len(batch_data)
                input_ids = torch.tensor(batch_data, dtype=torch.long).to(args.device)
                out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
                embs = out.last_hidden_state.cpu().float().numpy()
                embs = (embs - latent_mean) / latent_std
                shard_embs[i:i+bs] = embs.astype(np.float16)

        # Save shard
        shard_path = os.path.join(shards_dir, f"shard_{shard_idx:04d}.npy")
        np.save(shard_path, shard_embs)
        del shard_embs

        if (shard_idx + 1) % 5 == 0 or shard_idx == n_shards - 1:
            elapsed = time.time() - t0
            done = shard_idx + 1
            rate = (done * args.shard_size) / elapsed
            eta = (n_shards - done) * args.shard_size / rate if rate > 0 else 0
            print(f"  [{done}/{n_shards}] {rate:.0f} samples/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nEncoded {n_samples:,} samples in {elapsed:.0f}s ({n_samples/elapsed:.0f} samples/s)")

    # Save metadata
    meta = {
        "n_samples": n_samples,
        "seq_len": args.seq_len,
        "hidden_size": hidden_size,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "shard_size": args.shard_size,
        "n_shards": n_shards,
        "dtype": "float16",
        "model_name": args.model_name,
        "input_file": args.input,
    }
    meta_path = args.output + "_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput: {shards_dir}/ ({n_shards} shards)")
    print(f"Meta: {meta_path}")
    print("Done!")


if __name__ == "__main__":
    main()
