#!/usr/bin/env python3
"""Compare DNA tokenization strategies for ELF training."""

import os
import sys
import gzip
import random
import numpy as np
from collections import Counter
from pathlib import Path

# ─── 1. Extract sample DNA sequences from hg38 ───

def read_fasta_chunks(fasta_path, chunk_size=1024, max_chunks=5000):
    """Read non-overlapping chunks from FASTA."""
    chunks = []
    chrom = None
    seqs = []
    open_fn = gzip.open if fasta_path.endswith('.gz') else open
    
    with open_fn(fasta_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if chrom is not None and len(seqs) > 0:
                    full_seq = ''.join(seqs)
                    for i in range(0, len(full_seq) - chunk_size + 1, chunk_size):
                        chunk = full_seq[i:i + chunk_size]
                        if 'N' not in chunk:
                            chunks.append(chunk)
                            if len(chunks) >= max_chunks:
                                return chunks
                chrom = line[1:].split()[0]
                seqs = []
            else:
                seqs.append(line.upper())
        # Last chrom
        if seqs:
            full_seq = ''.join(seqs)
            for i in range(0, len(full_seq) - chunk_size + 1, chunk_size):
                chunk = full_seq[i:i + chunk_size]
                if 'N' not in chunk:
                    chunks.append(chunk)
                    if len(chunks) >= max_chunks:
                        return chunks
    return chunks


# ─── Tokenizers ───

class SingleNucleotideTokenizer:
    """Each nucleotide is one token."""
    VOCAB = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '<pad>': 0}
    
    def encode(self, seq):
        return [self.VOCAB.get(b, 0) for b in seq]
    
    def decode(self, ids):
        inv = {v: k for k, v in self.VOCAB.items()}
        return ''.join(inv.get(i, '') for i in ids if i > 0)
    
    @property
    def vocab_size(self):
        return len(self.VOCAB)


class DNABPETokenizer:
    """BPE tokenizer trained on DNA sequences."""
    def __init__(self, merges=None, vocab=None):
        self.merges = merges or {}
        self.vocab = vocab or {}
    
    @staticmethod
    def train(sequences, vocab_size=256, min_freq=2):
        """Train BPE on DNA sequences."""
        # Start with single nucleotides + special tokens
        vocab = {'<pad>': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4}
        next_id = len(vocab)
        
        # Count all pairs across sequences
        merge_order = []
        
        # Convert sequences to lists of tokens
        tokenized = [list(s) for s in sequences]
        
        for _ in range(vocab_size - 5):  # -5 for initial vocab
            # Count pairs
            pair_counts = Counter()
            for seq_tokens in tokenized:
                for i in range(len(seq_tokens) - 1):
                    pair_counts[(seq_tokens[i], seq_tokens[i+1])] += 1
            
            if not pair_counts:
                break
            
            best_pair = pair_counts.most_common(1)[0][0]
            if pair_counts[best_pair] < min_freq:
                break
            
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in vocab:
                vocab[merged_token] = next_id
                next_id += 1
                merge_order.append(best_pair)
            
            # Apply merge to all sequences
            for seq_tokens in tokenized:
                i = 0
                while i < len(seq_tokens) - 1:
                    if (seq_tokens[i], seq_tokens[i+1]) == best_pair:
                        seq_tokens[i] = merged_token
                        del seq_tokens[i+1]
                    else:
                        i += 1
        
        merges = {}
        for pair in merge_order:
            merges[pair] = pair[0] + pair[1]
        
        return DNABPETokenizer(merges=merges, vocab=vocab)
    
    def encode(self, seq):
        tokens = list(seq)
        for pair, merged in self.merges.items():
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(t, 0) for t in tokens]
    
    @property
    def vocab_size(self):
        return len(self.vocab)


def analyze_tokenization(name, tokenizer, sequences):
    """Analyze tokenization quality."""
    token_counts = []
    vocab_usage = Counter()
    reconstruct_errors = 0
    
    for seq in sequences:
        ids = tokenizer.encode(seq)
        token_counts.append(len(ids))
        for tid in ids:
            vocab_usage[tid] += 1
        
        # Check reconstruction
        if hasattr(tokenizer, 'decode'):
            decoded = tokenizer.decode(ids)
            if decoded != seq:
                reconstruct_errors += 1
    
    lengths = np.array(token_counts)
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Vocab size:        {tokenizer.vocab_size}")
    print(f"  Unique tokens used: {len(vocab_usage)}")
    print(f"  Token length:")
    print(f"    Mean:   {lengths.mean():.1f}")
    print(f"    Median: {np.median(lengths):.1f}")
    print(f"    Min:    {lengths.min()}")
    print(f"    Max:    {lengths.max()}")
    print(f"    Std:    {lengths.std():.1f}")
    print(f"  Compression ratio: {1024 / lengths.mean():.2f}x (bp per token)")
    print(f"  Reconstruction:    {'Perfect ✅' if reconstruct_errors == 0 else f'{reconstruct_errors} errors ❌'}")
    
    # Top 10 most common tokens
    if hasattr(tokenizer, 'VOCAB'):
        inv = {v: k for k, v in tokenizer.VOCAB.items()}
        top = vocab_usage.most_common(10)
        print(f"  Top tokens: {[(inv.get(tid, f'id={tid}'), cnt) for tid, cnt in top[:5]]}")
    elif hasattr(tokenizer, 'vocab'):
        inv = {v: k for k, v in tokenizer.vocab.items()}
        top = vocab_usage.most_common(10)
        print(f"  Top tokens: {[(inv.get(tid, f'id={tid}')[:10], cnt) for tid, cnt in top[:5]]}")
    
    return {
        'name': name,
        'vocab_size': tokenizer.vocab_size,
        'mean_tokens': lengths.mean(),
        'compression': 1024 / lengths.mean(),
        'reconstruction_ok': reconstruct_errors == 0,
    }


def main():
    fasta_path = "/home/stark/data/hg38.fa.gz"
    
    print("Loading DNA sequences from hg38...")
    sequences = read_fasta_chunks(fasta_path, chunk_size=1024, max_chunks=10000)
    print(f"Loaded {len(sequences)} sequences of 1024bp")
    
    # Verify they're all 1024bp
    assert all(len(s) == 1024 for s in sequences)
    
    results = []
    
    # ─── Test 1: T5 Tokenizer (baseline) ───
    print("\n--- Testing T5 tokenizer ---")
    from transformers import T5Tokenizer
    t5 = T5Tokenizer.from_pretrained('t5-small')
    
    class T5Wrapper:
        def __init__(self, tokenizer):
            self.t5 = tokenizer
            self._vocab_size = len(tokenizer)
        def encode(self, seq):
            return self.t5.encode(seq)
        def decode(self, ids):
            return self.t5.decode(ids)
        @property
        def vocab_size(self):
            return self._vocab_size
    
    results.append(analyze_tokenization("T5-small tokenizer (current)", T5Wrapper(t5), sequences[:1000]))
    
    # ─── Test 2: Single Nucleotide ───
    print("\n--- Testing Single Nucleotide tokenizer ---")
    sn = SingleNucleotideTokenizer()
    results.append(analyze_tokenization("Single Nucleotide (A/C/G/T)", sn, sequences))
    
    # ─── Test 3: BPE trained on DNA ───
    print("\n--- Training BPE tokenizer on DNA ---")
    train_seqs = sequences[:8000]
    test_seqs = sequences[8000:10000]
    
    for target_vocab in [32, 64, 128, 256, 512]:
        bpe = DNABPETokenizer.train(train_seqs, vocab_size=target_vocab)
        results.append(analyze_tokenization(
            f"DNA BPE (vocab={bpe.vocab_size})", bpe, test_seqs
        ))
    
    # ─── Summary ───
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'Vocab':>6} {'Tokens':>8} {'Comp.':>8} {'Recon.':>8}")
    print(f"{'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"{r['name']:<30} {r['vocab_size']:>6} {r['mean_tokens']:>8.1f} {r['compression']:>7.2f}x {'✅' if r['reconstruction_ok'] else '❌':>8}")
    
    # ─── NucEL feasibility analysis ───
    print(f"\n{'='*60}")
    print(f"  NucEL ENCODER ANALYSIS")
    print(f"{'='*60}")
    print("""
NucEL (BERT-based DNA encoder):
  - Input: single nucleotide (A/C/G/T/N) → 6 vocab tokens
  - Hidden dim: 768 (BERT-base architecture)
  - Max input length: 512 tokens (BERT positional embedding limit)
  - Pre-trained on: human genome + multi-species
  
  ⚠️ PROBLEM: BERT's max position embedding = 512
     Our sequences are 1024bp → need to truncate or extend positions
     
  For ELF integration:
  - NucEL outputs (batch, seq_len, 768) — same shape as T5-small's (batch, seq_len, 512)
  - ELF-B hidden_size=768 already — actually a BETTER match than T5's 512!
  - But: need to handle the 512→1024 length mismatch
  
  Options:
  a) Use NucEL with 512bp chunks (halves context)
  b) Extend NucEL position embeddings (requires fine-tuning the encoder)
  c) Use overlapping windows + pooling
""")


if __name__ == '__main__':
    main()
