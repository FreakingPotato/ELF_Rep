"""
Dataset and DataLoader for sharded NucEL embeddings.
"""
import json
import os
from typing import Optional

import jax
import numpy as np
import jax.numpy as jnp
from torch.utils.data import Dataset, DataLoader, Sampler
from flax.training.common_utils import shard


class ShardedEmbeddingDataset(Dataset):
    """Dataset that loads sharded .npy embedding files."""

    def __init__(self, shards_dir: str):
        self.shards = sorted([
            f for f in os.listdir(shards_dir) if f.endswith('.npy')
        ])
        self.shards_dir = shards_dir
        self._index = []  # list of (shard_file, local_idx)

        first = np.load(os.path.join(shards_dir, self.shards[0]), mmap_mode='r')
        self.hidden_size = first.shape[2]
        self.seq_len = first.shape[1]

        for shard_file in self.shards:
            path = os.path.join(shards_dir, shard_file)
            shard = np.load(path, mmap_mode='r')
            for i in range(len(shard)):
                self._index.append((shard_file, i))
            del shard

        self.n_samples = len(self._index)
        self._cached_shard = None
        self._cached_file = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        shard_file, local_idx = self._index[idx]
        if self._cached_file != shard_file:
            path = os.path.join(self.shards_dir, shard_file)
            self._cached_shard = np.load(path, mmap_mode='r')
            self._cached_file = shard_file
        emb = self._cached_shard[local_idx].astype(np.float32)
        return emb


class RandomBatchSampler(Sampler):
    """Yields batches of indices, shuffled, dropping last incomplete batch."""
    def __init__(self, dataset_len, batch_size, shuffle=True):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(self.dataset_len))
        if self.shuffle:
            np.random.shuffle(indices)
        # Drop last incomplete batch
        n_complete = (len(indices) // self.batch_size) * self.batch_size
        indices = indices[:n_complete]
        for i in range(0, len(indices), self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return self.dataset_len // self.batch_size


def collate_fn(indices_list):
    """indices_list is actually a list of indices from BatchSampler."""
    # This shouldn't be called with BatchSampler properly configured
    pass


def load_embedding_dataset(data_prefix: str):
    shards_dir = data_prefix + "_shards"
    meta_path = data_prefix + "_meta.json"
    assert os.path.isdir(shards_dir), f"Missing shards dir: {shards_dir}"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    dataset = ShardedEmbeddingDataset(shards_dir)
    return dataset, meta


def get_embedding_dataloader(
    data_prefix: str,
    batch_size: int,
    num_devices: int = 2,
):
    """Create a DataLoader that yields JAX-ready sharded batches directly."""
    dataset, meta = load_embedding_dataset(data_prefix)
    seq_len = dataset.seq_len
    hidden_size = dataset.hidden_size

    # Ensure batch_size is divisible by num_devices
    if batch_size % num_devices != 0:
        batch_size = (batch_size // num_devices) * num_devices
        if batch_size == 0:
            batch_size = num_devices

    class BatchIterator:
        def __init__(self):
            self.indices = list(range(len(dataset)))
            np.random.shuffle(self.indices)
            # Drop incomplete
            n = (len(self.indices) // batch_size) * batch_size
            self.indices = self.indices[:n]
            self.pos = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.pos >= len(self.indices):
                raise StopIteration
            idx_batch = self.indices[self.pos:self.pos + batch_size]
            self.pos += batch_size

            # Load batch
            embs = np.stack([dataset[i] for i in idx_batch])
            masks = np.ones((len(idx_batch), seq_len), dtype=np.float32)

            # Convert to JAX and shard
            batch = {
                "embeddings": jnp.array(embs),
                "attention_mask": jnp.array(masks),
            }
            return shard(batch)

        def __len__(self):
            return len(self.indices) // batch_size

    def dataloader():
        return BatchIterator()

    # Compute steps
    n_samples = (len(dataset) // batch_size) * batch_size
    steps_per_epoch = n_samples // batch_size

    return dataloader, meta, steps_per_epoch
