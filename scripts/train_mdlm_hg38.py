#!/usr/bin/env python3
"""MDLM (Masked Diffusion Language Model) baseline on hg38 DNA data.

Adapts the MDLM framework (Sahoo et al., NeurIPS 2024) to DNA sequence generation.
Uses the same NucEL token vocabulary (vocab_size=27) and hg38 1024bp data as ELF-B.

Architecture: DiT (Diffusion Transformer) matching ELF-B compute budget (~87M params)
- 12 layers, hidden=768, 12 heads, MLP ratio=4
- Absorbing state diffusion with SUBS parameterization
- Ancestral sampling with 128/256/512 steps

Usage:
  python scripts/train_mdlm_hg38.py \
    --config configs/training_configs/train_hg38_nucel_ELF-B.yml \
    --output outputs/mdlm-hg38/
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---- Dataset ----

class DNADataset(Dataset):
    """Load uint16 token sequences from binary file."""

    TOKEN_MAP = {11: 0, 12: 1, 13: 2, 14: 3}  # Map to 0-3 for MDLM

    def __init__(self, bin_path, seq_len=1024, max_samples=None):
        data = np.fromfile(bin_path, dtype=np.uint16)
        n = data.shape[0] // seq_len
        data = data[:n * seq_len].reshape(n, seq_len)
        # Filter to pure nucleotide sequences
        good = np.mean((data >= 11) & (data <= 14), axis=1) == 1.0
        data = data[good]
        if max_samples and len(data) > max_samples:
            idx = np.random.default_rng(42).choice(len(data), max_samples, replace=False)
            data = data[idx]
        # Map tokens: 11->0, 12->1, 13->2, 14->3
        self.data = np.vectorize(self.TOKEN_MAP.get)(data).astype(np.int64)
        self.seq_len = seq_len
        # Mask token = vocab_size (4 for DNA)
        self.mask_token = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


# ---- Noise Schedule ----

class LogLinearSchedule:
    """Log-linear noise schedule for absorbing state diffusion.

    sigma(t) = 1 - t, so the mask probability at time t is 1 - t.
    """
    def sigma(self, t):
        return 1.0 - t

    def sigma_inverse(self, sigma):
        return 1.0 - sigma


# ---- Model: DiT for discrete tokens ----

class DiTBlock(nn.Module):
    """Diffusion Transformer block with adaptive layer norm."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )
        # AdaLN modulation
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x, c):
        shift1, scale1, shift2, scale2, gate1, gate2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + gate2.unsqueeze(1) * self.mlp(h)
        return x


class DiTDNA(nn.Module):
    """DiT for DNA discrete diffusion.

    Architecture matches ELF-B compute: 12 layers, 768 hidden, 12 heads ~ 87M params.
    """

    def __init__(self, vocab_size=5, seq_len=1024, hidden_size=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, num_classes=1000):
        super().__init__()
        self.vocab_size = vocab_size  # 4 nucleotides + 1 mask
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # Token embedding (includes mask token)
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        # Position embedding
        self.pos_emb = nn.Embedding(seq_len, hidden_size)
        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        # Output head
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.head = nn.Linear(hidden_size, vocab_size - 1)  # Predict original 4 tokens

        self._init_weights()

    def _init_weights(self):
        # Scale embeddings
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, x, t):
        """x: [B, L] token ids (0-3 for nucleotides, 4 for mask). t: [B] time in [0,1]."""
        B, L = x.shape
        # Embed tokens + positions
        h = self.token_emb(x) + self.pos_emb(torch.arange(L, device=x.device))
        # Time embedding via sinusoidal
        t_emb = self._timestep_embed(t)  # [B, hidden]
        c = self.time_emb(t_emb)  # [B, hidden]
        # Transformer blocks
        for block in self.blocks:
            h = block(h, c)
        # Output
        h = self.norm(h)
        logits = self.head(h)  # [B, L, vocab_size-1]
        return logits

    def _timestep_embed(self, t, dim=256):
        """Sinusoidal timestep embedding."""
        half = dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) * (np.log(10000) / half))
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ---- SUBS Loss (MDLM) ----

def subs_loss(model, x, schedule):
    """MDLM SUBS loss: simplified absorbing state diffusion loss.

    At time t, randomly mask tokens with probability sigma(t) = 1-t.
    Loss is cross-entropy on masked positions predicting original tokens.
    """
    B, L = x.shape
    device = x.device

    # Sample t ~ U(0, 1)
    t = torch.rand(B, device=device)

    # Mask probability: sigma(t) = 1 - t
    mask_prob = schedule.sigma(t)  # [B]
    mask_prob_expanded = mask_prob[:, None].expand(B, L)  # [B, L]

    # Create mask
    mask = torch.rand(B, L, device=device) < mask_prob_expanded  # True = masked

    # Apply mask: replace with mask_token (4)
    x_noised = x.clone()
    x_noised[mask] = 4  # mask token

    # Predict
    logits = model(x_noised, t)  # [B, L, 4]

    # Loss: only on masked positions
    loss = F.cross_entropy(
        logits[mask].reshape(-1, 4),
        x[mask].reshape(-1),
    )
    return loss


# ---- Sampling ----

@torch.no_grad()
def sample(model, schedule, n_samples, seq_len, n_steps, device):
    """Ancestral sampling from MDLM.

    Start from all-mask, progressively denoise.
    """
    model.eval()
    x = torch.full((n_samples, seq_len), 4, dtype=torch.long, device=device)  # all mask

    for step in tqdm(range(n_steps), desc="Sampling"):
        t = 1.0 - step / n_steps  # t goes from 1 -> 0
        t_tensor = torch.full((n_samples,), t, device=device)
        t_next = max(0.0, 1.0 - (step + 1) / n_steps)

        # Which positions to denoise at this step
        sigma_t = schedule.sigma(t_tensor)  # [B]
        sigma_next = schedule.sigma(torch.full((n_samples,), t_next, device=device))

        # Number of positions to unmask
        n_mask = (x == 4).sum(dim=1)  # [B] currently masked
        n_unmask = ((sigma_t - sigma_next) * seq_len).long().clamp(min=0)  # [B]

        if n_unmask.sum() == 0:
            continue

        # Predict
        logits = model(x, t_tensor)  # [B, L, 4]
        probs = F.softmax(logits, dim=-1)

        # For each sequence, unmask top-k most confident positions
        confidence = probs.max(dim=-1).values  # [B, L]

        for b in range(n_samples):
            if n_unmask[b] == 0:
                continue
            mask_pos = (x[b] == 4).nonzero(as_tuple=True)[0]
            if len(mask_pos) == 0:
                continue
            k = min(n_unmask[b].item(), len(mask_pos))
            # Get confidence for masked positions
            conf = confidence[b, mask_pos]
            _, topk_idx = conf.topk(k)
            unmask_positions = mask_pos[topk_idx]
            # Sample from predicted distribution
            samples = torch.multinomial(probs[b, unmask_positions], 1).squeeze(-1)
            x[b, unmask_positions] = samples

    return x


# ---- Training ----

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/home/stark/.cache/dna-diffusion/nucel_data/train_1024.bin")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="outputs/mdlm-hg38/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    print("Loading data...")
    dataset = DNADataset(args.data, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)
    print(f"  {len(dataset)} sequences")

    # Model (~87M params)
    model = DiTDNA(
        vocab_size=5, seq_len=args.seq_len,
        hidden_size=args.hidden_size, depth=args.depth, num_heads=args.num_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MDLM-DiT parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = LogLinearSchedule()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device)
            loss = subs_loss(model, batch, schedule)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, time={elapsed:.0f}s")

        # Save checkpoint
        ckpt_path = os.path.join(args.output, f"checkpoint_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)
        print(f"  Saved {ckpt_path}")

    # Sample
    print("\nSampling 1000 sequences...")
    model.load_state_dict(torch.load(ckpt_path)["model"])
    gen = sample(model, schedule, 1000, args.seq_len, 128, device)

    # Convert back to DNA
    id2tok = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    seqs = []
    for row in gen:
        seq = ''.join(id2tok.get(t.item(), 'N') for t in row)
        seqs.append(seq)

    out_path = os.path.join(args.output, "generated_sequences.txt")
    with open(out_path, 'w') as f:
        for s in seqs:
            f.write(s + '\n')
    print(f"Saved {len(seqs)} sequences to {out_path}")

    # Print sample
    print(f"Sample: {seqs[0][:80]}...")
    gc = [(s.count('G') + s.count('C')) / len(s) for s in seqs]
    print(f"GC: mean={np.mean(gc):.4f} std={np.std(gc):.4f}")


if __name__ == "__main__":
    train()
