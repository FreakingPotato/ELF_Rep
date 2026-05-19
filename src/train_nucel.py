#!/usr/bin/env python3
"""Train ELF on precomputed NucEL embeddings.

Unlike train.py (which uses T5 encoder on-the-fly), this script loads
precomputed embeddings from disk — no PyTorch dependency during training.

Usage:
  python src/train_nucel.py --config configs/training_configs/train_hg38_nucel_ELF-B.yml
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
import yaml
from functools import partial

import jax
try:
    jax.distributed.initialize()
except (RuntimeError, ValueError):
    pass

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import jax.numpy as jnp
import numpy as np
from flax import jax_utils
from flax.training.common_utils import shard
from tqdm import tqdm

from modules.model import ELF_models
from configs.config import load_config_from_yaml, apply_config_overrides
from utils.train_utils import TrainState, prefetch_to_device, get_optimizer, create_learning_rate_fn
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.embedding_data_utils import get_embedding_dataloader
from train_step_precomputed import train_step_precomputed

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
for _name in ("absl", "orbax", "tensorstore", "flax.training.checkpoints"):
    logging.getLogger(_name).setLevel(logging.ERROR)
sys.stdout.reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--config_override", action="append", default=[])
    return parser.parse_args()


def run_training(config):
    log_for_0("=" * 60)
    log_for_0("ELF + NucEL Training (Precomputed Embeddings)")
    log_for_0("=" * 60)
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Data: {config.data_path}")
    log_for_0(f"Max seq len: {config.max_length}")
    log_for_0(f"Encoder dim: {config.encoder_model_name}")  # repurposed for hidden_size info
    log_for_0(f"JAX devices: {jax.device_count()} x {jax.default_backend()}")
    log_for_0("=" * 60)

    # Load metadata
    meta_path = config.data_path + "_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        encoder_dim = meta["hidden_size"]
        latent_mean = meta.get("latent_mean", 0.0)
        latent_std = meta.get("latent_std", 1.0)
        log_for_0(f"Dataset meta: {meta['n_samples']:,} samples, hidden={encoder_dim}")
    else:
        raise ValueError(f"Missing meta file: {meta_path}")

    # Override config latent stats from preprocessing
    config.latent_mean = latent_mean
    config.latent_std = latent_std
    log_for_0(f"Latent mean={latent_mean:.6f}, std={latent_std:.6f}")

    rng = jax.random.PRNGKey(config.seed)

    # Create ELF model
    log_for_0(f"Creating {config.model}...")
    model_fn = ELF_models[config.model]
    rng, init_rng, dropout_rng = jax.random.split(rng, 3)

    input_dim = 2 * encoder_dim if config.self_cond_prob > 0 else encoder_dim
    dummy_x = jnp.ones((1, config.max_length, input_dim))
    dummy_t = jnp.ones((1,))

    model = model_fn(
        text_encoder_dim=encoder_dim,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=27,  # NucEL vocab (no CE head needed)
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )

    init_args = dict(
        x=dummy_x, t=dummy_t, deterministic=True,
        self_cond_cfg_scale=jnp.ones((1,)) if config.num_self_cond_cfg_tokens > 0 else None,
    )
    elf_params = model.init(init_rng, **init_args)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(elf_params))
    log_for_0(f"ELF parameters: {total_params:,}")

    num_devices = jax.device_count()
    num_local_devices = jax.local_device_count()

    if config.global_batch_size is not None:
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // jax.process_count()
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        total_batch_size = config.batch_size * num_devices
        local_batch_size = config.batch_size * num_local_devices
    else:
        raise ValueError("Must specify global_batch_size or batch_size")

    # Create dataloader
    train_dataloader_fn, _, steps_per_epoch = get_embedding_dataloader(
        config.data_path,
        batch_size=local_batch_size,
        num_devices=num_devices,
    )
    num_train_steps = steps_per_epoch * config.epochs

    if config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    else:
        num_warmup_steps = 0

    grad_accum_steps = config.grad_accum_steps
    num_optimizer_steps = num_train_steps // grad_accum_steps
    num_warmup_optimizer_steps = num_warmup_steps // grad_accum_steps

    if config.lr is None or config.lr <= 0:
        config.lr = config.blr * (total_batch_size * grad_accum_steps) / 256

    log_for_0(
        f"batch={local_batch_size}, total={total_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total={num_train_steps}, lr={config.lr:.2e}"
    )

    lr_schedule = create_learning_rate_fn(
        num_train_steps=num_optimizer_steps,
        num_warmup_steps=num_warmup_optimizer_steps,
        learning_rate=config.lr,
        schedule=config.lr_schedule,
        min_lr=config.min_lr,
    )
    optimizer = get_optimizer(config, lr_schedule, grad_accum_steps=grad_accum_steps)
    state = TrainState.create(
        apply_fn=model.apply,
        params=elf_params["params"],
        tx=optimizer,
        dropout_rng=dropout_rng,
        ema_params1=copy.deepcopy(elf_params["params"]),
    )

    # Resume
    if not config.resume:
        auto_ckpt = find_latest_checkpoint(config.output_dir)
        if auto_ckpt:
            config.resume = config.output_dir
            log_for_0(f"Auto-resuming from {auto_ckpt}")

    start_epoch, resume_step = 0, 0
    if config.resume:
        try:
            ckpt_path = config.resume
            if "checkpoint_" not in ckpt_path:
                ckpt_path = find_latest_checkpoint(ckpt_path) or ckpt_path
            state, resume_step = load_checkpoint(ckpt_path, state)
            start_epoch = 0  # TODO: read from checkpoint metadata
            log_for_0(f"Resumed from step {resume_step}")
        except Exception as e:
            log_for_0(f"Checkpoint error: {e}. Starting fresh.")

    state = jax_utils.replicate(state)
    p_train_step = jax.pmap(
        partial(train_step_precomputed, config=config),
        axis_name="batch", donate_argnums=(0,),
    )

    os.makedirs(config.output_dir, exist_ok=True)
    config_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
    config_path = os.path.join(config.output_dir, 'config.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # Training loop
    log_for_0("\n" + "=" * 60)
    log_for_0("Starting Training")
    log_for_0("=" * 60)

    global_step = resume_step
    last_log_step = global_step
    train_metrics = []
    last_log_time = time.time()

    for epoch in range(start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")
        train_iter = train_dataloader_fn()

        pbar = tqdm(train_iter, total=steps_per_epoch,
                     desc=f"Epoch {epoch+1}", mininterval=1.0)

        for batch in pbar:
            # batch is already sharded JAX arrays
            state, metrics = p_train_step(state, batch=batch)

            train_metrics.append(metrics)
            global_step += 1

            if global_step % config.log_freq == 0:
                from flax.training.common_utils import get_metrics
                gathered = get_metrics(train_metrics)
                avg_loss = float(jnp.mean(gathered["loss"]))
                avg_l2 = float(jnp.mean(gathered["l2_loss"]))
                now = time.time()
                steps_per_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = lr_schedule((global_step - 1) // grad_accum_steps)

                bases_per_sec = steps_per_sec * total_batch_size * config.max_length
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    l2=f"{avg_l2:.4f}",
                    lr=f"{current_lr:.2e}",
                    sps=f"{steps_per_sec:.1f}",
                )
                log_for_0(
                    f"[{global_step}] loss={avg_loss:.4f} l2={avg_l2:.4f} "
                    f"lr={current_lr:.2e} steps/s={steps_per_sec:.1f} "
                    f"bases/s={bases_per_sec:,.0f}"
                )
                train_metrics = []
                last_log_step = global_step
                last_log_time = now

            if config.save_freq > 0 and global_step % int(config.save_freq) == 0:
                save_path = os.path.join(config.output_dir, f"checkpoint_{global_step}")
                save_checkpoint(state, save_path, global_step)
                log_for_0(f"Saved checkpoint: {save_path}")

        # Epoch tracking is done via the loop variable, no need to update state

    # Final save
    save_path = os.path.join(config.output_dir, f"checkpoint_{global_step}")
    save_checkpoint(state, save_path, global_step)
    log_for_0(f"\nTraining complete! Final checkpoint: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config_from_yaml(args.config)
    config = apply_config_overrides(config, args.config_override)
    run_training(config)
