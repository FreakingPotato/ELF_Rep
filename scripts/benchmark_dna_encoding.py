#!/usr/bin/env python3
"""
Actual benchmark of DNA encoding strategies for ELF training.
Tests: NucEL (k=1), DNA BPE, T5-small on real hg38 data.
Measures: tokenization stats, encoder output quality, encoder speed, memory.
"""

import os
import sys
import gzip
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ─── Load real DNA sequences from hg38 ───

def read_fasta_chunks(fasta_path, chunk_size=1024, max_chunks=5000):
    chunks = []
    chrom = None
    seqs = []
    open_fn = gzip.open if fasta_path.endswith('.gz') else open
    with open_fn(fasta_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if chrom is not None and seqs:
                    full_seq = ''.join(seqs)
                    for i in range(0, len(full_seq) - chunk_size + 1, chunk_size):
                        c = full_seq[i:i + chunk_size]
                        if 'N' not in c:
                            chunks.append(c)
                            if len(chunks) >= max_chunks:
                                return chunks
                chrom = line[1:].split()[0]
                seqs = []
            else:
                seqs.append(line.upper())
        if seqs:
            full_seq = ''.join(seqs)
            for i in range(0, len(full_seq) - chunk_size + 1, chunk_size):
                c = full_seq[i:i + chunk_size]
                if 'N' not in c:
                    chunks.append(c)
                    if len(chunks) >= max_chunks:
                        return chunks
    return chunks


def analyze_tokenization(name, token_lists, seqs):
    """Analyze tokenization quality from pre-computed token lists."""
    lengths = np.array([len(t) for t in token_lists])
    
    # Check reconstruction (decode tokens back to DNA)
    recon_ok = 0
    for i, (toks, seq) in enumerate(zip(token_lists[:100], seqs[:100])):
        # We check by re-encoding the decoded string
        recon_ok += 1  # Will be verified per-tokenizer below
    
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Sequences tested:    {len(token_lists)}")
    print(f"  Input length:        1024 bp each")
    print(f"  Token length:")
    print(f"    Mean:              {lengths.mean():.1f}")
    print(f"    Median:            {np.median(lengths):.1f}")
    print(f"    Min:               {lengths.min()}")
    print(f"    Max:               {lengths.max()}")
    print(f"    Std:               {lengths.std():.1f}")
    print(f"  Compression:         {1024 / lengths.mean():.2f} bp/token")
    
    return {
        'name': name,
        'mean_tokens': lengths.mean(),
        'std_tokens': lengths.std(),
        'compression': 1024 / lengths.mean(),
    }


def benchmark_encoder(name, encode_fn, sequences, n_warmup=5, n_runs=50):
    """Benchmark encoder forward pass speed and measure output statistics."""
    import torch
    
    print(f"\n  Encoder benchmark: {name}")
    
    # Warmup
    for i in range(n_warmup):
        with torch.no_grad():
            _ = encode_fn(sequences[i])
    
    # Measure
    times = []
    outputs = []
    for i in range(n_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        with torch.no_grad():
            out = encode_fn(sequences[i])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if i < 10:
            outputs.append(out)
    
    times = np.array(times)
    
    # Output statistics
    all_outs = np.concatenate([o.flatten() for o in outputs])
    
    print(f"    Time per sequence: {times.mean()*1000:.1f} ± {times.std()*1000:.1f} ms")
    print(f"    Throughput:        {1.0/times.mean():.1f} seq/sec")
    print(f"    Output shape:      {outputs[0].shape}")
    print(f"    Output mean:       {all_outs.mean():.4f}")
    print(f"    Output std:        {all_outs.std():.4f}")
    print(f"    Output min:        {all_outs.min():.4f}")
    print(f"    Output max:        {all_outs.max():.4f}")
    
    # Measure peak GPU memory
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"    GPU memory peak:   {mem:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    return {
        'name': name,
        'time_ms': times.mean() * 1000,
        'throughput': 1.0 / times.mean(),
        'output_shape': tuple(outputs[0].shape),
        'output_mean': float(all_outs.mean()),
        'output_std': float(all_outs.std()),
    }


def main():
    fasta_path = "/home/stark/data/hg38.fa.gz"
    
    print("=" * 70)
    print("  DNA ENCODING BENCHMARK FOR ELF TRAINING")
    print("  Testing on real hg38 data")
    print("=" * 70)
    
    # Load sequences
    print("\nLoading hg38 sequences...")
    sequences = read_fasta_chunks(fasta_path, chunk_size=1024, max_chunks=5000)
    print(f"  Loaded {len(sequences)} sequences of 1024bp")
    
    token_results = []
    encoder_results = []
    
    # ═══════════════════════════════════════════════════════
    # TEST 1: NucEL (k=1, actual config)
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  TEST 1: NucEL Encoder (k=1, ModernBERT)")
    print("─" * 70)
    
    import torch
    nucel_path = "/home/stark/.cache/huggingface/hub/models--FreakingPotato--NucEL/snapshots/723b30c4d09bd3ae9ed8d02426ad5cf806ea8659/"
    sys.path.insert(0, nucel_path)
    from tokenizer import NucEL_Tokenizer
    
    # Use ACTUAL config: k=1
    nucel_tok = NucEL_Tokenizer(k=1)
    print(f"  NucEL tokenizer: k={nucel_tok.k}, vocab_size={nucel_tok.vocab_size}")
    print(f"  Vocab: {nucel_tok.get_vocab()}")
    
    # Tokenize
    nucel_token_lists = []
    for seq in sequences:
        ids = nucel_tok.encode(seq)
        nucel_token_lists.append(ids)
    
    # Check reconstruction
    nucel_recon_errors = 0
    for i, (toks, seq) in enumerate(zip(nucel_token_lists[:1000], sequences[:1000])):
        decoded = nucel_tok.decode(toks, skip_special_tokens=True).replace(' ', '')
        if decoded != seq:
            nucel_recon_errors += 1
            if nucel_recon_errors <= 3:
                print(f"    Recon error #{nucel_recon_errors}: decoded[:50]={decoded[:50]} vs seq[:50]={seq[:50]}")
    print(f"  Reconstruction: {1000-nucel_recon_errors}/1000 perfect ({nucel_recon_errors} errors)")
    
    tr1 = analyze_tokenization("NucEL (k=1)", nucel_token_lists, sequences)
    token_results.append({**tr1, 'reconstruction': f"{1000-nucel_recon_errors}/1000"})
    
    # NucEL encoder (need newer transformers for ModernBERT)
    try:
        from transformers import AutoModel, AutoTokenizer
        nucel_model = AutoModel.from_pretrained("FreakingPotato/NucEL")
        nucel_model.eval()
        if torch.cuda.is_available():
            nucel_model = nucel_model.cuda()
        
        total_params = sum(p.numel() for p in nucel_model.parameters())
        print(f"  NucEL params: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Hidden size: {nucel_model.config.hidden_size}")
        print(f"  Max positions: {nucel_model.config.max_position_embeddings}")
        print(f"  Num layers: {nucel_model.config.num_hidden_layers}")
        
        def nucel_encode(seq):
            toks = nucel_tok(seq, return_tensors='pt', truncation=True, max_length=2048)
            if torch.cuda.is_available():
                toks = {k: v.cuda() for k, v in toks.items()}
            out = nucel_model(**toks)
            return out.last_hidden_state[0].cpu().numpy()
        
        er1 = benchmark_encoder("NucEL (k=1)", nucel_encode, sequences)
        encoder_results.append(er1)
    except Exception as e:
        print(f"  ⚠️ Cannot load NucEL model: {e}")
        print(f"  (Need transformers >= 4.50 for ModernBERT)")
    
    # ═══════════════════════════════════════════════════════
    # TEST 2: DNA BPE Tokenizer (trained on hg38)
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  TEST 2: DNA BPE Tokenizer (trained on hg38)")
    print("─" * 70)
    
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers import models, trainers, pre_tokenizers
    
    train_seqs = sequences[:4000]
    test_seqs = sequences[4000:5000]
    
    bpe_results = {}
    for target_vs in [64, 256, 1024, 4096]:
        print(f"\n  Training BPE vocab_size={target_vs}...")
        bpe_tok = HFTokenizer(models.BPE())
        bpe_tok.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=target_vs,
            special_tokens=['<pad>', '<unk>', '<cls>', '<sep>', '<mask>'],
            initial_alphabet=['A', 'C', 'G', 'T'],
            min_frequency=5,
        )
        bpe_tok.train_from_iterator(train_seqs, trainer=trainer)
        
        actual_vs = bpe_tok.get_vocab_size()
        
        # Tokenize test set
        bpe_token_lists = []
        for seq in test_seqs:
            enc = bpe_tok.encode(seq)
            bpe_token_lists.append(enc.ids)
        
        # Check reconstruction
        bpe_recon_errors = 0
        for i, (toks, seq) in enumerate(zip(bpe_token_lists[:1000], test_seqs[:1000])):
            decoded = bpe_tok.decode(toks).replace(' ', '')
            if decoded != seq:
                bpe_recon_errors += 1
                if bpe_recon_errors <= 2:
                    print(f"    Recon error #{bpe_recon_errors}: decoded[:60]='{decoded[:60]}' vs seq[:60]='{seq[:60]}'")
        
        tr = analyze_tokenization(f"DNA BPE (vocab={actual_vs})", bpe_token_lists, test_seqs)
        token_results.append({**tr, 'reconstruction': f"{1000-bpe_recon_errors}/1000"})
        
        # Save tokenizer for later use
        bpe_results[actual_vs] = {
            'tokenizer': bpe_tok,
            'token_lists': bpe_token_lists,
        }
    
    # ═══════════════════════════════════════════════════════
    # TEST 3: T5-small (current baseline)
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  TEST 3: T5-small (current baseline)")
    print("─" * 70)
    
    from transformers import T5Tokenizer, T5EncoderModel
    
    t5_tok = T5Tokenizer.from_pretrained('t5-small', legacy=False)
    t5_model = T5EncoderModel.from_pretrained('t5-small')
    t5_model.eval()
    if torch.cuda.is_available():
        t5_model = t5_model.cuda()
    
    total_params = sum(p.numel() for p in t5_model.parameters())
    print(f"  T5 params: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Tokenize
    t5_token_lists = []
    for seq in sequences:
        ids = t5_tok.encode(seq)
        t5_token_lists.append(ids)
    
    # Check reconstruction
    t5_recon_errors = 0
    for i, (toks, seq) in enumerate(zip(t5_token_lists[:1000], sequences[:1000])):
        decoded = t5_tok.decode(toks, skip_special_tokens=True)
        if decoded != seq:
            t5_recon_errors += 1
            if t5_recon_errors <= 3:
                print(f"    Recon error #{t5_recon_errors}: decoded[:50]={decoded[:50]} vs seq[:50]={seq[:50]}")
    print(f"  Reconstruction: {1000-t5_recon_errors}/1000 perfect ({t5_recon_errors} errors)")
    
    tr3 = analyze_tokenization("T5-small (current)", t5_token_lists, sequences)
    token_results.append({**tr3, 'reconstruction': f"{1000-t5_recon_errors}/1000"})
    
    # T5 encoder benchmark
    def t5_encode(seq):
        toks = t5_tok(seq, return_tensors='pt', truncation=True, max_length=1024)
        if torch.cuda.is_available():
            toks = {k: v.cuda() for k, v in toks.items()}
        out = t5_model(**toks)
        return out.last_hidden_state[0].cpu().numpy()
    
    er3 = benchmark_encoder("T5-small", t5_encode, sequences)
    encoder_results.append(er3)
    
    # ═══════════════════════════════════════════════════════
    # TEST 4: Analyze what BPE tokens look like
    # ═══════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  ANALYSIS: BPE Token Structure")
    print("─" * 70)
    
    for vs, data in list(bpe_results.items())[:3]:
        tok = data['tokenizer']
        vocab = tok.get_vocab()
        # Group tokens by length
        len_dist = Counter()
        for token in vocab:
            if not token.startswith('<'):
                len_dist[len(token)] += 1
        print(f"\n  BPE vocab={vs}: token length distribution")
        for length in sorted(len_dist.keys()):
            print(f"    Length {length}: {len_dist[length]} tokens")
    
    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<25} {'Vocab':>6} {'Tokens/seq':>11} {'Comp.':>7} {'Recon.':>10}")
    print("-" * 70)
    for r in token_results:
        vs_str = ""
        print(f"{r['name']:<25} {vs_str:>6} {r['mean_tokens']:>8.1f}±{r['std_tokens']:.1f} {r['compression']:>6.2f}x {r.get('reconstruction','?'):>10}")
    
    if encoder_results:
        print(f"\n{'Encoder':<25} {'Time/seq':>10} {'seq/sec':>10} {'Output shape':>20}")
        print("-" * 70)
        for r in encoder_results:
            print(f"{r['name']:<25} {r['time_ms']:>8.1f}ms {r['throughput']:>10.1f} {str(r['output_shape']):>20}")


if __name__ == '__main__':
    main()
