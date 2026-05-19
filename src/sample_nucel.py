#!/usr/bin/env python3
"""
Sample DNA sequences from a trained ELF-B + NucEL model.

Usage:
  python src/sample_nucel.py \
    --checkpoint outputs/elf-b-hg38-nucel/checkpoint_48870 \
    --config configs/training_configs/train_hg38_nucel_ELF-B.yml \
    --n_samples 10000 --sampling_steps 200 --batch_size 8 \
    --output results/generated/
"""
import argparse, json, os, sys, time
import jax, jax.numpy as jnp, numpy as np
from flax import jax_utils
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.model import ELF_models


# ---- Config loader ----
def load_config(yaml_path):
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    class C: pass
    c = C()
    for k, v in cfg.items():
        setattr(c, k, v)
    for k, d in [('t_eps',1e-4),('denoiser_p_mean',-1.2),('denoiser_p_std',1.2),
                 ('time_schedule','logit_normal'),('denoiser_noise_scale',1.0),
                 ('self_cond_prob',0.0),('num_self_cond_cfg_tokens',0),
                 ('bottleneck_dim',32),('latent_mean',0.0),('latent_std',1.0),
                 ('attn_dropout',0.0),('proj_dropout',0.0),('num_model_mode_tokens',0)]:
        if not hasattr(c, k): setattr(c, k, d)
    return c


def make_model(config):
    return ELF_models[config.model](
        text_encoder_dim=512, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=27, num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
    )


def decode_to_dna(embeddings, emb_matrix, token_map={11:'A',12:'C',13:'G',14:'T'}):
    """Nearest-neighbour decode: embeddings [B,L,512] -> list of DNA strings."""
    B, L, H = embeddings.shape
    flat = embeddings.reshape(-1, H)
    # Cosine similarity to nucleotide tokens only (11-14)
    tok_emb = emb_matrix[11:15]  # [4, 512]
    flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
    tok_n = tok_emb / (np.linalg.norm(tok_emb, axis=1, keepdims=True) + 1e-8)
    sims = flat_n @ tok_n.T  # [N, 4]
    ids = np.argmax(sims, axis=1) + 11  # [N]
    seqs = []
    for i in range(B):
        tokens = ids[i*L:(i+1)*L]
        seqs.append(''.join(token_map.get(t, 'N') for t in tokens))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--sampling_steps", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--output", default="results/generated/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"{'='*60}\nELF-B + NucEL Sampling\n{'='*60}")
    config = load_config(args.config)
    seq_len, H = config.max_length, 512
    ndev = jax.device_count()
    bs = max(ndev, (args.batch_size // ndev) * ndev)  # align to devices

    # Model + params
    model = make_model(config)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, seq_len, H)), jnp.ones((1,)), deterministic=True)

    import orbax.checkpoint as ocp
    ckpt_path = os.path.abspath(args.checkpoint)
    if os.path.isdir(os.path.join(ckpt_path, os.path.basename(ckpt_path))):
        ckpt_path = os.path.join(ckpt_path, os.path.basename(ckpt_path))
    ckpt = ocp.StandardCheckpointer().restore(ckpt_path)
    ema = ckpt['ema_params1']
    print(f"Checkpoint step {ckpt.get('step','?')} loaded. Devices: {ndev}")

    # Build pmap'd denoise function
    t_eps = config.t_eps
    def _denoise_scan(z_shard, t_steps_shard):
        """z_shard: [dev_batch, seq, 512], t_steps: [steps+1]"""
        def step(z, tp):
            t, tn = tp[0], tp[1]
            t_b = jnp.full((z.shape[0],), t)
            net_out = model.apply({'params': ema}, z, t_b, deterministic=True)[0]
            t_r = t.reshape(1,1,1)
            v = (net_out - z) / jnp.maximum(1.0 - t_r, t_eps)
            return z + (tn - t) * v, None
        t_pairs = jnp.stack([t_steps_shard[:-1], t_steps_shard[1:]], axis=1)
        z_f, _ = jax.lax.scan(step, z_shard, t_pairs)
        return z_f

    p_denoise = jax.pmap(_denoise_scan, axis_name='batch')

    # Shard helpers
    def make_t_steps(rng):
        return jnp.linspace(0.0, 1.0, args.sampling_steps + 1)  # uniform ODE

    def make_noise(rng, total_batch):
        per = total_batch // ndev
        rngs = jax.random.split(rng, ndev)
        return jnp.stack([jax.random.normal(rngs[i], (per, seq_len, H))
                          * config.denoiser_noise_scale for i in range(ndev)])

    # Decode matrix
    emb_matrix = np.load("data/nucel_embedding_matrix.npy")

    # Meta for normalization
    meta_path = config.data_path + "_meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f: meta = json.load(f)
        lmean, lstd = meta.get('latent_mean',0), meta.get('latent_std',1)
    else:
        lmean, lstd = 0.0, 1.0
    print(f"latent mean={lmean:.4f} std={lstd:.4f}")

    os.makedirs(args.output, exist_ok=True)
    rng = jax.random.PRNGKey(args.seed)
    t_steps = jnp.broadcast_to(make_t_steps(rng), (ndev, args.sampling_steps + 1))

    all_seqs = []
    nb = (args.n_samples + bs - 1) // bs
    t0 = time.time()

    for bi in tqdm(range(nb), desc="Sampling"):
        cur = min(bs, args.n_samples - len(all_seqs))
        cur = max(ndev, ((cur + ndev - 1) // ndev) * ndev)
        rng, nr = jax.random.split(rng)
        z = make_noise(nr, cur)
        z_f = p_denoise(z, t_steps)
        emb = np.array(z_f.reshape(-1, seq_len, H))
        emb = emb * lstd + lmean  # denormalize
        seqs = decode_to_dna(emb, emb_matrix)
        all_seqs.extend(seqs[:min(cur, args.n_samples - len(all_seqs) + cur)])

    all_seqs = all_seqs[:args.n_samples]
    elapsed = time.time() - t0
    print(f"\n{len(all_seqs)} seqs in {elapsed:.1f}s ({len(all_seqs)/elapsed:.1f} seq/s)")

    # Save FASTA + txt
    with open(os.path.join(args.output, "generated.fasta"), 'w') as f:
        for i, s in enumerate(all_seqs):
            f.write(f">seq_{i}\n{s}\n")
    with open(os.path.join(args.output, "generated_sequences.txt"), 'w') as f:
        for s in all_seqs: f.write(s + "\n")
    json.dump({"n":len(all_seqs),"seq_len":seq_len,"steps":args.sampling_steps,
               "checkpoint":args.checkpoint,"mean":lmean,"std":lstd},
              open(os.path.join(args.output, "sampling_meta.json"),'w'), indent=2)

    gc = [(s.count('G')+s.count('C'))/max(len(s),1) for s in all_seqs]
    print(f"GC mean={np.mean(gc):.4f} std={np.std(gc):.4f}")
    print(f"Unique: {len(set(all_seqs))}/{len(all_seqs)}")
    print(f"Sample: {all_seqs[0][:80]}...")


if __name__ == "__main__":
    main()
