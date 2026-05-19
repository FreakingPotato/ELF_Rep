# ELF-B + NucEL DNA Generation Model — Evaluation Plan

## 参考论文及核心指标

| 论文 | 发表 | 核心评估方法 |
|------|------|-------------|
| **DNA-Diffusion** (Pinello Lab, Nature Genetics 2025) | 染色质可及性 | Motif频率、BLAT比对、多样性、Cell-type specificity预测 |
| **DiscDiff** (ICML 2024) | Latent Diffusion | S-FID、Motif频率相关性、FReD (Fréchet Reconstruction Distance) |
| **DDSM** (Avdeyev et al., Nature Biotech 2023) | Dirichlet扩散 | NLL、GC含量、k-mer分布、Motif位置频率 |
| **EvoDiff** (Microsoft, 2023) | 离散扩散 | 序列novelty、多样性、结构合理性 |

---

## 1. 采样 (Sampling)

从训练好的模型生成 ~10,000 条 DNA 序列用于评估。

```bash
# 需要写的采样脚本
python src/sample_nucel.py \
  --checkpoint outputs/elf-b-hg38-nucel/checkpoint_48870 \
  --config configs/training_configs/train_hg38_nucel_ELF-B.yml \
  --n_samples 10000 \
  --seq_len 1024 \
  --steps 200 \
  --output results/generated_samples/
```

采样流程：
1. 从 N(0,1) 生成噪声 (shape: [10000, 1024, 512])
2. ELF-B denoiser iteratively denoise (200 steps)
3. 用 NucEL decoder 将 latent → DNA tokens
4. Token IDs → ATCG strings

---

## 2. 定量评估指标 (Quantitative Metrics)

### 2.1 S-FID (Sequence Fréchet Inception Distance) — [DiscDiff]
- **方法**: 用 NucEL encoder 分别编码 real 和 generated sequences 为 embeddings
- 计算 embedding 均值向量和协方差矩阵
- FID = ||μ_real - μ_gen||² + Tr(Σ_real + Σ_gen - 2(Σ_real Σ_gen)^{1/2})
- **越低越好**，衡量分布距离

### 2.2 Motif 频率相关性 (Motif Frequency Correlation) — [DiscDiff, DDSM]
- 在 real 和 generated 序列中扫描关键 motif 的位置分布
- 计算每个位置上 motif 出现频率的 Pearson correlation
- 关键 motifs: TATA-box (TATAAA), Initiator (CCAAT), GC-box (GGGCGG), CCAAT-box
- **越高越好** (接近 1.0)

### 2.3 FReD (Fréchet Reconstruction Distance) — [DiscDiff]
- 用 VAE/AutoEncoder 的 latent space 计算 FID
- 我们可以用 NucEL 的 latent space 直接计算
- **越低越好**

### 2.4 GC Content 分布 — [DDSM, DNA-Diffusion]
- 计算 real vs generated 序列的 GC 含量直方图
- 用 Wasserstein distance 或 KS test 量化差异
- **分布越接近越好**

### 2.5 K-mer 分布 — [通用]
- 计算 k=3,4,5,6 的 k-mer 频率分布
- Real vs Generated 的 cosine similarity
- **越高越好**

### 2.6 序列 Novelty (Uniqueness) — [DNA-Diffusion]
- BLAST/BLAT 将生成序列与训练集比对
- 计算 average alignment length
- 越短说明越 novel（不是复制训练数据）
- 同时计算内部重复率（生成序列之间的相似度）

### 2.7 序列多样性 (Diversity) — [EvoDiff, DNA-Diffusion]
- 生成序列两两之间的平均 edit distance / Hamming distance
- 与训练集的 diversity 对比
- **应该接近训练集的 diversity**

### 2.8 Nucleotide Composition — [通用]
- A/T/G/C 频率分布对比
- Dinucleotide 频率对比 (16 个 dinucleotides)
- KS test / Wasserstein distance

---

## 3. 可视化 (Visualizations)

### 3.1 t-SNE / UMAP Embedding 可视化
- 用 NucEL encoder 编码 real + generated sequences
- t-SNE / UMAP 降维到 2D
- 不同颜色标记 real vs generated
- **判断**: 两个分布是否重叠？generated 是否覆盖了 real 的多样性？

### 3.2 Motif 位置频率热图 (Positional Motif Frequency)
- X轴: 序列位置 (0-1024)
- Y轴: 不同 models (Real, ELF-B, 随机基线)
- 热图颜色: 特定 motif 在每个位置的频率
- 类似 DiscDiff Figure 1 的 TATA-box 位置分布图

### 3.3 Loss / FID 采样曲线
- 在不同 denoising steps (10, 25, 50, 100, 200, 500) 采样
- 计算每个 step 的 S-FID
- 画 S-FID vs denoising steps 曲线
- 验证模型收敛行为

### 3.4 GC Content / K-mer 分布直方图
- Overlay histogram: real (蓝色) vs generated (橙色)
- 包含: GC content, 4-mer 频率 top-20, dinucleotide 频率

### 3.5 染色质可及性预测 (Chromatin Accessibility) — [DNA-Diffusion]
- 用预训练的 Enformer/Sei chromatin predictor
- 预测生成序列的染色质 profile
- 与 real DHS sequences 的 profile 对比

---

## 4. 基线对比 (Baselines)

| 方法 | 说明 |
|------|------|
| **Random** | 随机生成 ATCG 序列，匹配 GC content |
| **Markov** | 从训练集学习 k-mer transition matrix，生成序列 |
| **GAN** | 如果有 DNA-GAN baseline |
| **ELF-B (ours)** | 我们训练的模型 |
| **Real data** | 真实 chr21 DHS sequences |

---

## 5. 实现优先级

### Phase 1: 采样 + 基础评估 (1-2天)
1. 编写采样脚本 (`src/sample_nucel.py`)
2. 生成 10,000 条序列
3. 实现 S-FID、GC content、nucleotide composition
4. 实现 k-mer 分布对比

### Phase 2: Motif 分析 + 可视化 (1-2天)
5. Motif 扫描和位置频率
6. t-SNE/UMAP 可视化
7. 分布直方图

### Phase 3: 高级评估 (2-3天)
8. BLAT novelty 分析
9. 染色质可及性预测 (如果有 Enformer)
10. Denoising steps 曲线
11. 与 baseline 对比，生成论文级表格和图

---

## 6. 代码结构

```
ELF/
├── src/
│   ├── sample_nucel.py          # 采样脚本
│   └── evaluate/
│       ├── metrics.py           # S-FID, FReD, GC, k-mer
│       ├── motif_analysis.py    # Motif 频率和位置分析
│       ├── novelty.py           # BLAT 比对和多样性
│       └── visualize.py         # 所有可视化
├── results/
│   ├── generated_samples/       # 生成的序列
│   ├── metrics/                 # 评估结果 JSON
│   └── figures/                 # 可视化图表
└── EVALUATION_PLAN.md
```
