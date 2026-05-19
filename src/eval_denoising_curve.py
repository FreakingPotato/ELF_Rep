#!/usr/bin/env python3
"""
T8: Denoising Steps Curve — simplified single-device version.
"""
import sys, os, json, time
import jax, jax.numpy as jnp, numpy as np
import orbax.checkpoint as ocp

# Force single device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.model import ELF_models


def load_config(yaml_path):
    import yaml
    with open(yaml_path) as f: cfg = yaml.safe_load(f)
    class C: pass
    c = C()
    for k, v in cfg.items(): setattr(c, k, v)
    for k, d in [('t_eps',1e-4),('denoiser_noise_scale',1.0),('self_cond_prob',0.0),
                 ('num_self_cond_cfg_tokens',0),('bottleneck_dim',32),('latent_mean',0.0),
                 ('latent_std',1.0),('attn_dropout',0.0),('proj_dropout',0.0),
                 ('num_model_mode_tokens',0)]:
        if not hasattr(c, k): setattr(c, k, d)
    return c


def decode(embeddings, emb_matrix, H=512):
    B, L, _ = embeddings.shape
    flat = embeddings.reshape(-1, H)
    tok = emb_matrix[11:15]
    flat_n = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8)
    tok_n = tok / (np.linalg.norm(tok, axis=1, keepdims=True) + 1e-8)
    ids = np.argmax(flat_n @ tok_n.T, axis=1) + 11
    tmap = {11:'A',12:'C',13:'G',14:'T'}
    return [''.join(tmap.get(t,'N') for t in ids[i*L:(i+1)*L]) for i in range(B)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default="results/evaluation/figures")
    ap.add_argument("--n_per_step", type=int, default=64)
    ap.add_argument("--steps_list", default="5,10,25,50,100")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)
    seq_len, H = config.max_length, 512

    print("Creating model...")
    model = ELF_models[config.model](
        text_encoder_dim=512, max_length=seq_len, attn_drop=0, proj_drop=0,
        num_time_tokens=config.num_time_tokens, num_self_cond_cfg_tokens=0,
        vocab_size=27, num_model_mode_tokens=0, bottleneck_dim=config.bottleneck_dim)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, seq_len, H)), jnp.ones((1,)), deterministic=True)

    ckpt_path = os.path.abspath(args.checkpoint)
    if os.path.isdir(os.path.join(ckpt_path, os.path.basename(ckpt_path))):
        ckpt_path = os.path.join(ckpt_path, os.path.basename(ckpt_path))
    ckpt = ocp.StandardCheckpointer().restore(ckpt_path)
    ema = ckpt['ema_params1']
    print(f"Checkpoint loaded (step {ckpt.get('step','?')})")

    with open(config.data_path + '_meta.json') as f: meta = json.load(f)
    lmean, lstd = meta['latent_mean'], meta['latent_std']
    emb_matrix = np.load("data/nucel_embedding_matrix.npy")

    # Compile denoise function
    t_eps = config.t_eps

    @jax.jit
    def denoise_jit(z, t_steps):
        def step(z, tp):
            t, tn = tp[0], tp[1]
            net_out = model.apply({'params': ema}, z, jnp.full((z.shape[0],), t), deterministic=True)[0]
            t_r = t.reshape(1, 1, 1)
            v = (net_out - z) / jnp.maximum(1.0 - t_r, t_eps)
            return z + (tn - t) * v, None
        return jax.lax.scan(step, z, jnp.stack([t_steps[:-1], t_steps[1:]], axis=1))[0]

    # Warmup JIT
    print("JIT warmup...")
    _ = denoise_jit(jnp.ones((2, seq_len, H)), jnp.linspace(0, 1, 6))
    _.block_until_ready()
    print("JIT ready.")

    # Load real data
    from evaluate_generated import load_real_sequences
    real_seqs = load_real_sequences("/home/stark/.cache/dna-diffusion/nucel_data/train_1024.bin", max_samples=1000)

    steps_list = [int(s) for s in args.steps_list.split(',')]
    n = args.n_per_step
    bs = 2  # 2 seqs per batch
    nb = n // bs

    results = []
    rng = jax.random.PRNGKey(42)

    for nsteps in steps_list:
        print(f"\nSteps = {nsteps}...")
        t_steps = jnp.linspace(0, 1, nsteps + 1)
        all_seqs = []
        t0 = time.time()

        for bi in range(nb):
            rng, nr = jax.random.split(rng)
            z = jax.random.normal(nr, (bs, seq_len, H))
            z_f = denoise_jit(z, t_steps)
            emb = np.array(z_f) * lstd + lmean
            all_seqs.extend(decode(emb, emb_matrix))

        elapsed = time.time() - t0

        # Metrics
        from evaluate.metrics import compare_gc_distribution, compare_kmer_distribution
        gc = compare_gc_distribution(real_seqs, all_seqs)
        km = compare_kmer_distribution(real_seqs, all_seqs)

        r = {
            "steps": nsteps,
            "time": round(elapsed, 1),
            "gc_wasserstein": gc["wasserstein_distance"],
            "gc_ks": gc["ks_statistic"],
            "kmer3_cosine": km["k=3"]["cosine_similarity"],
            "kmer4_cosine": km["k=4"]["cosine_similarity"],
        }
        results.append(r)
        print(f"  {elapsed:.1f}s  GC_W={r['gc_wasserstein']:.4f}  k3={r['kmer3_cosine']:.4f}  k4={r['kmer4_cosine']:.4f}")

    # Plot
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    steps = [r["steps"] for r in results]

    axes[0].plot(steps, [r["gc_wasserstein"] for r in results], 'o-', color='#4C72B0', lw=2, ms=8)
    axes[0].set_title('GC Wasserstein Distance', fontsize=13)
    axes[0].set_xlabel('Sampling Steps'); axes[0].set_ylabel('Distance ↓')

    axes[1].plot(steps, [r["kmer3_cosine"] for r in results], 'o-', color='#55A868', lw=2, ms=8)
    axes[1].set_title('3-mer Cosine Similarity', fontsize=13)
    axes[1].set_xlabel('Sampling Steps'); axes[1].set_ylabel('Similarity ↑')

    axes[2].plot(steps, [r["time"] for r in results], 'o-', color='#C44E52', lw=2, ms=8)
    axes[2].set_title('Wall-clock Time', fontsize=13)
    axes[2].set_xlabel('Sampling Steps'); axes[2].set_ylabel('Seconds')

    fig.suptitle('ELF-B: Quality vs Sampling Steps', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "denoising_steps.png"), dpi=150, bbox_inches='tight')
    print(f"\nSaved denoising_steps.png")

    json.dump(results, open(os.path.join(args.output, "denoising_steps.json"), 'w'), indent=2)
    print(json.dumps(results, indent=2))
    print("Done!")


if __name__ == "__main__":
    main()
