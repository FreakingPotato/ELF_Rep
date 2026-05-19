"""Per-device pmap'd training step for ELF with precomputed NucEL embeddings.

Unlike train_step.py (which encodes text on-the-fly with T5),
this version receives precomputed embeddings directly — no encoder needed.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from utils.train_utils import TrainState
from utils.sampling_utils import sample_timesteps, add_noise, net_out_to_v_x


Array = jnp.ndarray


def train_step_precomputed(
    state: TrainState,
    batch: Dict[str, Array],
    config,
) -> Tuple[TrainState, Dict[str, float]]:
    """Training step with precomputed embeddings (no encoder)."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std

    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale

    new_dropout_rng, current_step_rng = jax.random.split(state.dropout_rng, 2)
    current_step_rng = jax.random.fold_in(current_step_rng, jax.lax.axis_index(axis_name="batch"))

    (
        t_rng, noise_rng, self_cond_mask_rng, self_cond_cfg_rng, _,
        model_dropout_rng, decoder_step_rng, decoder_rng,
        decoder_lambda_rng, decoder_noise_rng, _,
    ) = jax.random.split(current_step_rng, 11)

    # Precomputed embeddings are already normalized (mean/std applied during preprocessing)
    x0 = batch["embeddings"]
    attention_mask = batch["attention_mask"]

    # For unconditional generation, no condition mask needed
    batch_size, seq_length = x0.shape[0], x0.shape[1]
    cond_seq_mask = jnp.zeros((batch_size, seq_length, 1))  # No conditioning tokens

    # Loss mask: all valid positions
    loss_mask = attention_mask

    t = sample_timesteps(
        t_rng, batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
    )

    noise = jax.random.normal(noise_rng, x0.shape, dtype=x0.dtype)

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)

    decoder_targets = batch.get("input_ids")  # May not exist for precomputed
    decoder_step_active = jax.random.bernoulli(decoder_step_rng, decoder_prob) if decoder_targets is not None else False

    # Decoder branch input
    if decoder_targets is not None:
        decoder_lambda_rng, decoder_noise_rng = jax.random.split(decoder_rng)
        decoder_z_vals = (
            jax.random.normal(decoder_lambda_rng, (batch_size * seq_length,))
            * config.decoder_p_std + config.decoder_p_mean
        )
        decoder_lambda_t = jax.nn.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
        decoder_noise = jax.random.normal(decoder_noise_rng, x0.shape, dtype=x0.dtype) * decoder_noise_scale
        decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / jnp.maximum(1 - t_expanded, t_eps)

    if self_cond_prob > 0:
        use_self_cond_mask = (
            (jax.random.uniform(self_cond_mask_rng, (batch_size,)) < self_cond_prob)
            .reshape(-1, 1, 1).astype(x0.dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        from utils.sampling_utils import sample_cfg_scale
        self_cond_cfg_scale = sample_cfg_scale(
            self_cond_cfg_rng, batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
        )
    else:
        self_cond_cfg_scale = None

    def get_z_input(params, z, t_input, self_cond_cfg_input, x_tokens):
        if self_cond_prob == 0:
            return z
        z_uncond = jnp.zeros_like(z)
        z_with_zeros = jnp.concatenate([z, z_uncond], axis=-1)
        net_out_init = state.apply_fn(
            {"params": params}, z_with_zeros, t_input,
            deterministic=True,
            self_cond_cfg_scale=self_cond_cfg_input,
        )
        net_out_init = jax.lax.stop_gradient(net_out_init)
        _, x_pred_init = net_out_to_v_x(net_out_init, z, t_input, t_eps)
        x_pred_cond = x_pred_init * use_self_cond_mask.astype(z.dtype)
        return jnp.concatenate([z, x_pred_cond], axis=-1)

    def reduce_token_loss(per_token_loss, loss_mask):
        loss_mask = loss_mask.astype(per_token_loss.dtype)
        safe_loss = jnp.where(loss_mask > 0, per_token_loss, jnp.zeros_like(per_token_loss))
        return (safe_loss * loss_mask).sum() / jnp.maximum(loss_mask.sum(), 1.0)

    def loss_fn(params):
        # Denoiser branch only (no decoder for precomputed embeddings initially)
        denoiser_input = get_z_input(
            params, denoiser_z, t,
            self_cond_cfg_input=self_cond_cfg_scale,
            x_tokens=x0,
        )
        net_out, _ = state.apply_fn(
            {"params": params}, denoiser_input, t,
            deterministic=False,
            rngs={"dropout": model_dropout_rng},
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=jnp.array(False),
        )
        v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, t_eps)

        per_dim_loss = (v_pred - v_target) ** 2
        l2_loss = reduce_token_loss(jnp.mean(per_dim_loss, axis=-1), loss_mask)
        return l2_loss, (l2_loss, jnp.zeros(()))

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (l2_loss_val, ce_loss_val)), grads = grad_fn(state.params)

    grads = jax.lax.pmean(grads, axis_name="batch")
    loss = jax.lax.pmean(loss, axis_name="batch")
    l2_loss_val = jax.lax.pmean(l2_loss_val, axis_name="batch")

    new_state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)

    def ema_update(ema_params, params, decay):
        return jax.tree_util.tree_map(lambda e, p: e * decay + p * (1 - decay), ema_params, params)

    is_optimizer_step = (new_state.step % config.grad_accum_steps) == 0
    new_ema_params1 = jax.lax.cond(
        is_optimizer_step,
        lambda: ema_update(state.ema_params1, new_state.params, config.ema_decay1),
        lambda: state.ema_params1,
    )
    new_state = new_state.replace(ema_params1=new_ema_params1, dropout_rng=new_dropout_rng)

    metrics = {
        "loss": loss,
        "l2_loss": l2_loss_val,
        "ce_loss": jnp.zeros(()),
    }
    return new_state, metrics
