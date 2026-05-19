#!/usr/bin/env python3
"""Extract and save the NucEL input embedding matrix for nearest-neighbor decoding.

Loads the NucEL model from HuggingFace, extracts its token embedding layer
(input_embeddings), and saves as a NumPy .npy file.

Output: data/nucel_embedding_matrix.npy — shape [27, 512]
  - Row i corresponds to NucEL token ID i
  - DNA nucleotides: 11=A, 12=C, 13=G, 14=T

Usage:
  python scripts/build_nucel_emb_matrix.py [--output data/nucel_embedding_matrix.npy]
"""
import argparse
import os

import numpy as np
import torch
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser(description="Extract NucEL embedding matrix")
    parser.add_argument(
        "--model", default="FreakingPotato/NucEL",
        help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output", default="data/nucel_embedding_matrix.npy",
        help="Output .npy path"
    )
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    # Extract the token embedding layer (nn.Embedding)
    emb_layer = model.get_input_embeddings()
    weight = emb_layer.weight.detach().cpu().numpy()  # [vocab_size, hidden_size]

    print(f"Embedding matrix shape: {weight.shape}")
    print(f"  Vocabulary size: {weight.shape[0]}")
    print(f"  Hidden dimension: {weight.shape[1]}")

    # Verify nucleotide tokens are distinguishable
    dna_tokens = {11: 'A', 12: 'C', 13: 'G', 14: 'T'}
    for i, (tid, name) in enumerate(dna_tokens.items()):
        emb = weight[tid]
        emb_n = emb / (np.linalg.norm(emb) + 1e-8)
        for j, (tid2, name2) in enumerate(dna_tokens.items()):
            if j > i:
                emb2 = weight[tid2]
                emb2_n = emb2 / (np.linalg.norm(emb2) + 1e-8)
                cos = float(emb_n @ emb2_n)
                dist = float(np.linalg.norm(emb - emb2))
                print(f"  {name}-{name2}: cosine={cos:.4f}, L2={dist:.4f}")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.save(args.output, weight)
    print(f"Saved to {args.output} ({os.path.getsize(args.output) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
