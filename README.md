# ELF-Rep: ELF for DNA Sequence Generation

> 基于 [ELF (Embedded Language Flows)](https://arxiv.org/abs/2605.10938) 的 DNA 序列生成复现与扩展。将 ELF flow-matching 框架从自然语言迁移到基因组序列生成，使用 NucEL 预训练编码器将 DNA token 映射到连续 embedding 空间进行扩散建模。

[![arXiv](https://img.shields.io/badge/ELF-arXiv%202605.10938-b31b1b.svg)](https://arxiv.org/abs/2605.10938)
[![NucEL](https://img.shields.io/badge/Encoder-NucEL-blue.svg)](https://huggingface.co/FreakingPotato/NucEL)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚠️ 与原始 ELF-B 的差异

本实现是 ELF-B 的 **DNA 领域变体**，与论文中的 ELF-B (105M 参数) 存在以下差异：

| 参数 | Paper ELF-B | 本实现 | 说明 |
|------|-------------|--------|------|
| Parameters | 105M | **86.7M** | 更小的 bottleneck |
| `bottleneck_dim` | 128 | **32** | Text projection 瓶颈层更小 |
| `self_cond_prob` | 0.5 | **0.0** | 未使用 self-conditioning |
| `decoder_prob` | 0.5 | **0.0** | 纯 L2 denoising，无 CE decoder head |
| `num_time_tokens` | 4 | **2** | 更少的时间步 prefix tokens |
| Optimizer | Muon | **AdamW** | 不同优化器 |

这些简化是为了适配 DNA 序列的 embedding space 特性（512 维 vs T5 的 768 维）和 GPU 显存限制。

---

## 📋 Overview

本项目将 ELF（连续 embedding 空间的 flow-matching 扩散模型）应用于 **hg38 人类基因组 DNA 序列生成**：

- **编码器**: NucEL（预训练 DNA 语言模型，512 维 embedding）
- **架构**: ELF-B（12 层 Transformer，86.7M 参数）
- **训练数据**: hg38 基因组，1024bp 窗口，预计算 NucEL embeddings
- **训练**: 10 epochs，48,870 steps，~30h (2×RTX 3090)，global batch size = 8
- **采样**: Euler ODE solver，50 步

## 🧬 Evaluation Results

详细报告见 [EVALUATION_REPORT.md](EVALUATION_REPORT.md) 或 [EVALUATION_REPORT.html](EVALUATION_REPORT.html)。

| Metric | ELF-B | Random Baseline | 判断 |
|--------|-------|-----------------|------|
| **S-FID** | **8.25** | — | ✅ |
| **Motif Pearson r** | **0.921** | — | ✅ 强相关 |
| **GC Wasserstein** | **0.023** | 0.102 | ✅ 4.5× 优于 random |
| **3-mer cosine** | **0.947** | 0.872 | ✅ |
| **4-mer cosine** | **0.904** | 0.822 | ✅ |
| **5-mer cosine** | **0.848** | 0.770 | ✅ |
| **Dinucleotide cosine** | **0.974** | — | ✅ |
| **Diversity (Hamming)** | **0.737** | 0.740 (real) | ✅ ≈ real |
| **Unique ratio** | **100%** | — | ✅ |
| **Novelty (20-mer)** | **99.9%** | — | ✅ |

### Denoising Steps 分析

| Steps | GC Wasserstein ↓ | 3-mer Cosine ↑ | Time |
|-------|-------------------|-----------------|------|
| 5 | 0.057 | 0.922 | 7s |
| 10 | 0.040 | 0.940 | 19s |
| 25 | 0.029 | 0.938 | 29s |
| **50** | **0.022** | **0.946** | **49s** |
| 100 | 0.010 | 0.935 | 94s |

---

## 🚀 Quick Start

### 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# JAX with CUDA:
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 数据预处理

将 hg38 基因组序列转为 NucEL embeddings：

```bash
python scripts/preprocess_nucel_embeddings.py \
  --fasta_path data/hg38.fa \
  --seq_len 1024 \
  --output_dir data/nucel_data/
```

输出：
- `train_1024.bin` — uint16 token IDs
- `train_1024_embeddings.bin` — float32 NucEL embeddings
- `train_1024_meta.json` — latent mean/std 统计

### 训练

```bash
cd src/
python train_nucel.py \
  --config configs/training_configs/train_hg38_nucel_ELF-B.yml
```

配置文件位于 `configs/training_configs/train_hg38_nucel_ELF-B.yml`。

### 采样

```bash
python src/sample_nucel.py \
  --checkpoint outputs/elf-b-hg38-nucel/checkpoint_48870 \
  --config configs/training_configs/train_hg38_nucel_ELF-B.yml \
  --n_samples 10000 --sampling_steps 50 --batch_size 8 \
  --output results/generated/
```

### 评估

```bash
# 基础指标 + 图表
python src/evaluate_generated.py \
  --generated results/generated/generated_sequences.txt \
  --real_data /path/to/train_1024.bin \
  --output results/evaluation/

# S-FID + t-SNE（需要 GPU + transformers）
python src/eval_sfid_tsne.py --output results/evaluation/

# Denoising steps 曲线
CUDA_VISIBLE_DEVICES=0 python src/eval_denoising_curve.py \
  --checkpoint outputs/elf-b-hg38-nucel/checkpoint_48870 \
  --config configs/training_configs/train_hg38_nucel_ELF-B.yml \
  --output results/evaluation/figures
```

### MDLM Baseline 对比

```bash
# 构建 NucEL embedding matrix（采样和评估需要）
python scripts/build_nucel_emb_matrix.py

# 训练 MDLM baseline (~87M params, 同等计算预算)
python scripts/train_mdlm_hg38.py \
  --data /path/to/train_1024.bin \
  --epochs 10 --batch_size 32 --lr 1e-4 \
  --output outputs/mdlm-hg38/

# DNA PPL 评估（NucEL MLM pseudo-perplexity）
python scripts/eval_dna_ppl.py \
  --sequences results/generated/generated_sequences.txt \
  --sequences outputs/mdlm-hg38/generated_sequences.txt \
  --labels ELF-B MDLM \
  --real_data /path/to/train_1024.bin \
  --output results/evaluation/dna_ppl.json
```

---

## 📁 Project Structure

```
ELF/
├── configs/
│   └── training_configs/
│       ├── train_hg38_nucel_ELF-B.yml   # DNA 生成训练配置
│       └── ...                           # 原始 ELF 文本任务配置
├── src/
│   ├── train_nucel.py                   # DNA 训练入口
│   ├── train_step_precomputed.py         # 训练 step（预计算 embedding）
│   ├── sample_nucel.py                  # DNA 序列采样
│   ├── evaluate_generated.py            # 评估 pipeline
│   ├── eval_sfid_tsne.py               # S-FID + t-SNE 计算
│   ├── eval_denoising_curve.py          # Denoising steps 曲线
│   ├── run_evaluation.py                # 完整评估 runner
│   ├── evaluate/
│   │   ├── metrics.py                   # 评估指标实现
│   │   └── visualize.py                 # 可视化生成
│   ├── utils/
│   │   ├── embedding_data_utils.py      # Embedding 数据加载器
│   │   ├── sampling_utils.py            # 采样工具函数
│   │   ├── generation_utils.py          # 生成工具函数
│   │   └── checkpoint_utils.py          # Checkpoint 加载
│   ├── modules/
│   │   ├── model.py                     # ELF/ELFBlock/TimestepEmbedder 等
│   │   └── layers.py                    # Attention, SwiGLUFFN, BottleneckTextProj 等
│   └── configs/
│       └── config.py                    # Config dataclass
├── scripts/
│   └── preprocess_nucel_embeddings.py   # 数据预处理脚本
├── EVALUATION_PLAN.md                   # 评估方案
├── EVALUATION_REPORT.md                 # 评估报告 (Markdown)
├── EVALUATION_REPORT.html               # 评估报告 (HTML)
└── README.md                            # 本文件
```

---

## 🔬 Technical Details

### Architecture

| Component | Detail |
|-----------|--------|
| Model | ELF-B: 12-layer Transformer |
| Hidden size | 768 |
| Attention heads | 12 |
| Bottleneck dim | 32 |
| Encoder | NucEL (frozen, hidden=512) |
| Parameters | 86,683,163 (~86.7M) |

### Flow Matching

ELF 使用连续时间 flow matching（非 DDPM）：
- **前向过程**: 数据 → 噪声，`x_t = (1-t) * x_0 + t * noise`
- **反向过程**: 噪声 → 数据，Euler ODE solver
- **Loss**: L2 回归预测 `v_t = (noise - x_0) / max(1-t, t_eps)`

### Decoding

NucEL 是 encoder-only 模型，没有 decoder。生成后通过 **cosine similarity 最近邻** 将 embedding 解码回 DNA token：

```python
# 生成 embedding [B, L, 512] → 最近邻到 NucEL token embeddings [4, 512]
# Token: 11=A, 12=C, 13=G, 14=T
sims = emb_normalized @ token_emb_normalized.T  # [N, 4]
token_ids = argmax(sims, axis=-1) + 11
```

---

## 📊 Comparison with Published DNA Diffusion Models

| Model | Method | S-FID | 4-mer cos | GC W-dist |
|-------|--------|-------|-----------|-----------|
| **ELF-B (ours)** | Flow-matching + NucEL | **8.25** | **0.904** | **0.023** |
| DiscDiff | Latent diffusion | ~5-15 | ~0.90 | ~0.02 |
| DNA-Diffusion | DDPM (one-hot) | N/A | ~0.85 | ~0.05 |
| DDSM | Dirichlet discrete | N/A | ~0.88 | ~0.03 |

> 注：不同论文使用不同数据集，数值仅供参考。

---

## 📝 Citation

**ELF 原始论文：**
```bib
@article{elf2026,
  title={ELF: Embedded Language Flows},
  author={Hu, Keya and Qiu, Linlu and Lu, Yiyang and Zhao, Hanhong and Li, Tianhong and Kim, Yoon and Andreas, Jacob and He, Kaiming},
  journal={arXiv preprint arXiv:2605.10938},
  year={2026}
}
```

**NucEL 编码器：**
```bib
@misc{nucel2025,
  title={NucEL: Nucleotide Embedding Language},
  author={FreakingPotato},
  howpublished={\url{https://huggingface.co/FreakingPotato/NucEL}},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
