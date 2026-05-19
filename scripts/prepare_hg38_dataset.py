#!/usr/bin/env python3
"""Prepare hg38 genome dataset for ELF training.

Splits the genome into non-overlapping 1024bp chunks, T5-tokenizes them,
and saves as HuggingFace Arrow dataset compatible with ELF's data loader.
"""

import os
import sys
import argparse
import gzip
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import T5Tokenizer


def read_fasta(fasta_path: str):
    """Yield (chrom, sequence) from a FASTA file."""
    chrom = None
    seqs = []
    open_fn = gzip.open if fasta_path.endswith('.gz') else open
    
    with open_fn(fasta_path, 'rt') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if chrom is not None:
                    yield chrom, ''.join(seqs)
                chrom = line[1:].split()[0]
                seqs = []
            else:
                seqs.append(line.upper())
        if chrom is not None:
            yield chrom, ''.join(seqs)


def chunk_sequence(seq: str, chunk_size: int = 1024):
    """Split sequence into non-overlapping chunks of chunk_size."""
    # Filter out sequences with N bases for clean training data
    chunks = []
    for i in range(0, len(seq) - chunk_size + 1, chunk_size):
        chunk = seq[i:i + chunk_size]
        if 'N' not in chunk:  # Skip chunks with ambiguous bases
            chunks.append(chunk)
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta', type=str, required=True, help='Path to hg38.fa or hg38.fa.gz')
    parser.add_argument('--output', type=str, default='data/hg38_t5_1024', help='Output directory')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Sequence chunk size')
    parser.add_argument('--max_chunks', type=int, default=None, help='Max number of chunks (for testing)')
    parser.add_argument('--chromosomes', type=str, nargs='*', default=None,
                        help='Only use these chromosomes (e.g., chr1 chr2)')
    args = parser.parse_args()

    print(f"Loading T5 tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    print(f"Reading FASTA: {args.fasta}")
    all_chunks = []
    chroms_seen = []
    
    for chrom, seq in read_fasta(args.fasta):
        if args.chromosomes and chrom not in args.chromosomes:
            continue
        chunks = chunk_sequence(seq, args.chunk_size)
        all_chunks.extend(chunks)
        chroms_seen.append((chrom, len(seq), len(chunks)))
        print(f"  {chrom}: {len(seq):,} bp -> {len(chunks):,} chunks")
    
    print(f"\nTotal chunks: {len(all_chunks):,}")
    
    if args.max_chunks:
        all_chunks = all_chunks[:args.max_chunks]
        print(f"Limited to {args.max_chunks} chunks for testing")
    
    # Tokenize
    print("Tokenizing with T5...")
    all_input_ids = []
    all_seq_lengths = []
    
    for i, chunk in enumerate(all_chunks):
        ids = tokenizer.encode(chunk)
        all_input_ids.append(ids)
        all_seq_lengths.append(len(ids))
        if (i + 1) % 100000 == 0:
            print(f"  Tokenized {i+1:,}/{len(all_chunks):,}")
    
    print(f"  Avg token length: {np.mean(all_seq_lengths):.1f}")
    print(f"  Max token length: {max(all_seq_lengths)}")
    print(f"  Min token length: {min(all_seq_lengths)}")
    
    # Create HuggingFace dataset
    print(f"Saving to {args.output}...")
    ds = Dataset.from_dict({
        'input_ids': all_input_ids,
        'sequence_length': all_seq_lengths,
    })
    
    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)
    
    print(f"Done! Dataset saved with {len(ds):,} samples")
    print(f"\nChromosome summary:")
    for chrom, seq_len, n_chunks in chroms_seen:
        print(f"  {chrom}: {seq_len:,} bp -> {n_chunks:,} chunks")


if __name__ == '__main__':
    main()
