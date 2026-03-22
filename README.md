# optogptv2

A transformer-based model for thin-film optical design, built on top of [optogpt](https://github.com/someone/optogpt). The key architectural change is a **dual-sequence representation**: materials and thicknesses are separated into two independent streams rather than fused into a single token. The thickness stream is treated as a **continuous float** (nm), not a categorical token.

---

## Quick Start

### 1. Install

```bash
uv sync
# or
pip install -e .
```

### 2. Generate data

```bash
uv run python generate_data.py \
    --n_samples 100000 --min_layers 1 --max_layers 10 \
    --dev_split 0.1 --val_split 0.1 \
    --out_dir ./data --nk_dir ./nk \
    --workers 32
```

Outputs `data/train.arrow`, `data/dev.arrow`, and `data/val.arrow`.

### 3. Train

**ForwardModel** (structure → spectrum):
```bash
uv run python train_forward.py \
    --train_path ./data/train.arrow --dev_path ./data/dev.arrow \
    --d_model 512 --n_layers 6 --n_heads 8 \
    --epochs 200 --batch_size 256 --run_name forward_v1
```

**InverseModel** (spectrum → structure):
```bash
uv run python train_inverse.py \
    --train_path ./data/train.arrow --dev_path ./data/dev.arrow \
    --d_model 512 --n_layers 6 --n_heads 8 \
    --epochs 200 --batch_size 256 --run_name inverse_v1
```

Checkpoints are saved to `saved_models/forward/` and `saved_models/inverse/` as `*_best.pt` (lowest dev loss) and `*_latest.pt` (every epoch).

### 4. Evaluate

**ForwardModel:**
```bash
uv run python evaluate.py \
    --checkpoint saved_models/forward/forward_v1_best.pt \
    --val_path ./data/val.arrow \
    --plot_dir ./plots/forward_eval
```

**InverseModel** (with TMM re-simulation):
```bash
uv run python evaluate.py \
    --checkpoint saved_models/inverse/inverse_v1_best.pt \
    --val_path ./data/val.arrow \
    --nk_dir ./nk --n_samples 1000 \
    --plot_dir ./plots/inverse_eval
```

Writes `metrics.json` (MSE, MAE, R²), scatter plot, loss curve, sample spectrum comparisons, and design comparison plots (inverse only) to the plot directory.

---

## Overview

Two models are provided:

| Model | Task | Architecture |
|---|---|---|
| **ForwardModel** | Structure → Spectrum | Encoder-only transformer |
| **InverseModel** | Spectrum → Structure | Single decoder + two output heads |

Both models share the dual-sequence representation. The ForwardModel predicts the optical spectrum from a given thin-film stack; the InverseModel autoregressively generates a thin-film stack that produces a target spectrum.

---

## Setup

```bash
# Install dependencies (Python 3.10+)
pip install -e .

# Or with uv
uv sync
```

Dependencies: `torch>=2.1`, `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `tmm`, `colour-science`.

The `nk/` directory must contain per-material CSV files with columns `wl` (wavelength in µm), `n`, and `k`.

---

## Data Generation

Thin-film structures are randomly sampled and simulated via the **Transfer Matrix Method (TMM)** using the `tmm` library.

### Design space

| Parameter | Range |
|---|---|
| Materials | 17 options (Al, Al2O3, AlN, Ge, HfO2, ITO, MgF2, MgO, Si, Si3N4, SiO2, Ta2O5, TiN, TiO2, ZnO, ZnS, ZnSe) + Glass\_Substrate substrate |
| Thicknesses | 5–250 nm, 5 nm steps (50 discrete values) |
| Layers per stack | 1–10 (configurable) |
| Wavelength range | 400–1100 nm, 10 nm steps (71 wavelengths) |
| Polarisation | s-polarisation, normal incidence |

Each sample is a random stack of 1–10 layers drawn from 17 materials with a Glass\_Substrate backing. Refractive index data (n+ik) is loaded from CSV files and cubic-spline interpolated to the simulation wavelengths.

### Output format

```json
{
  "materials":   [["Ta2O5", "AlN", "SiO2"], ...],
  "thicknesses": [[300.0, 320.0, 150.0], ...],
  "spectra":     [[0.12, 0.14, ..., 0.33, 0.41, ...], ...]
}
```

Each sample has a `materials` list of material name strings and a parallel `thicknesses` list of float values in nm. Each spectrum is a flat list of 142 floats: 71 reflectance values followed by 71 transmittance values, ordered by wavelength (400–1100 nm).

### Running data generation

```bash
python generate_data.py \
    --n_samples 100000 \
    --min_layers 1 \
    --max_layers 10 \
    --dev_split 0.1 \
    --val_split 0.1 \
    --out_dir ./data \
    --nk_dir ./nk \
    --workers 8 \
    --seed 42
```

Outputs `data/train.arrow`, `data/dev.arrow`, and `data/val.arrow`. Dev is used during training for early stopping; val is held out for final evaluation. Simulation is parallelised across workers using Python's `multiprocessing.Pool`; each worker loads the nk data once in its initialiser.

---

## Model Architecture

### Dual-sequence representation

The original optogpt fuses material and thickness into a single token (e.g. `Ta2O5_300`), requiring a large combined vocabulary (~900 tokens). optogptv2 separates the two streams:

```
Layer i:  material  →  "Ta2O5"   (categorical, vocab size ~20)
          thickness →   300.0    (continuous float, nm)
```

This has several advantages:
- The material vocabulary stays small (~20 tokens), making learning easier.
- Thickness is treated as a continuous value: the model can predict any nm value rather than choosing from 50 discrete buckets.
- "Ta2O5 at 300nm" and "Ta2O5 at 320nm" share a material embedding, giving better generalisation.

### Positional encoding: RoPE

Both architectures use **Rotary Position Embedding (RoPE)** (Su et al. 2021) instead of sinusoidal additive encoding. RoPE rotates Q and K vectors inside each attention head by a position-dependent angle, so:
- Token embeddings carry only semantic content (no positional noise).
- Attention scores naturally encode relative distance through the rotation difference.
- The positions vector is configurable — it defaults to `[0, 1, 2, ...]` but can be replaced with any physical quantity, which is exploited in Architecture B below.

Two architectures are planned and will be benchmarked against each other.

---

### Architecture A — Thickness as embedding (current)

Thickness is projected into the embedding space via a learned linear layer and added to the material embedding. RoPE uses standard sequential positions `[0, 1, 2, ...]`.

#### ForwardModel A

```
src_mat [B, S]  (long)   ─┐
                           ├─ DualEmbedding ─→ Encoder (N layers) ─→ CLS token ─→ SpectrumHead ─→ spectrum [B, W]
src_thk [B, S]  (float)  ─┘
```

**DualEmbedding** combines the two streams per position:

```
embed(i) = (Embedding_mat(mat_id[i]) + Linear_thk(thk_val[i] / 250)) × √d_model
```

Material tokens use `nn.Embedding`; thickness uses `nn.Linear(1, d_model)` applied to the normalised nm value. The representations are summed before entering the encoder. RoPE inside each attention layer uses positions `[0, 1, ..., S-1]`.

#### InverseModel A

```
spectrum [B, 1, W] ─→ SpectrumProjection ─→ memory [B, 1, d_model]
                                                  │
mat_ids  [B, S-1] ─┐                             │
                    ├─ DualEmbedding ─→ Decoder (cross-attn to memory) ─→ hidden [B, S-1, d_model]
thk_vals [B, S-1] ─┘                                                           │
                                                                   ┌────────────┴───────────┐
                                                                   │                        │
                                                               mat_head               thk_head
                                                         log P(mat) [B, S-1, V]    thk_preds [B, S-1] (nm)
```

The decoder input at each step is the DualEmbedding of the previous material token and the previous predicted thickness. Two output heads share the hidden state: `mat_head` (categorical) and `thk_head` (regression in nm).

---

### Architecture B — Thickness as RoPE position (planned)

Instead of projecting thickness into the embedding space, the nm value is used directly as the **position argument to RoPE** inside each attention layer. The embedding only carries material identity; the physical thickness determines how Q and K are rotated, so attention scores encode optical-depth-relative relationships rather than sequence-index-relative ones.

#### Key idea

In standard RoPE, position `i` means "the i-th token in sequence order". In Architecture B, position `i` is replaced by `thk_val[i]` (or a normalised/cumulative version), so the model reasons about layers at their physical nm depth rather than their index.

```
Standard RoPE:    positions = [0,   1,   2,   3  ]   (sequence index)
Architecture B:   positions = [80, 120, 200, 150 ]   (nm per layer)
  or cumulative:  positions = [80, 200, 400, 550 ]   (cumulative depth nm)
```

#### ForwardModel B

```
src_mat [B, S]  (long)   ─→ Embedding_mat × √d_model ─→ Encoder (RoPE uses thk_vals) ─→ CLS ─→ SpectrumHead ─→ spectrum [B, W]
src_thk [B, S]  (float)  ─────────────────────────────────────────────────────────────────────────────────────────^
                                                         (passed as positions to MultiHeadAttention, not embedded)
```

Changes from Architecture A:
- No `DualEmbedding` — only `Embedding_mat` (material-only)
- `src_thk` is passed as the `positions` argument to `MultiHeadAttention` at every encoder layer
- RoPE rotates Q and K by angles determined by the nm thickness of each layer

#### InverseModel B

Same change on the decoder side: the decoder embedding is material-only, and `thk_vals` are passed as RoPE positions. The output heads are unchanged (`mat_head` categorical, `thk_head` regression).

#### Hypothesis

If optical interactions (e.g. interference, resonance) depend primarily on physical layer thickness rather than stack order, Architecture B may learn better representations: two stacks with identical layer thicknesses but different ordering will produce different attention patterns in A (since they differ in sequence position) but the same in B (since RoPE positions are identical), letting the model focus on what matters physically.

---

### Shared building blocks

| Component | Detail |
|---|---|
| `MultiHeadAttention` | Named `w_q/w_k/w_v/w_o` projections; RoPE applied to Q and K only; accepts optional `positions` tensor |
| `FeedForward` | `Linear → GELU → Dropout → Linear` |
| `ResidualConnection` | Pre-norm: `x + dropout(sublayer(LayerNorm(x)))` |
| `Encoder` | N × EncoderLayer + final LayerNorm |
| `Decoder` | N × DecoderLayer (self-attn + cross-attn + FFN) + final LayerNorm |

Weight initialisation: Xavier uniform for all weight matrices (dim > 1).

### Default hyperparameters

| Parameter | Default |
|---|---|
| `d_model` | 512 |
| `d_ff` | 2048 |
| `n_heads` | 8 |
| `n_layers` | 6 |
| `dropout` | 0.1 |

---

## Training

### Learning rate schedule

**Cosine annealing with linear warmup:**

```
warmup:  linear ramp from ~0 → peak_lr over warmup_steps
decay:   cosine curve from peak_lr → min_lr over remaining steps
```

Default: `peak_lr=3e-4`, `warmup_steps=2000`, `min_lr=1e-6`.

### ForwardModel

Loss: `MSELoss` between predicted and ground-truth spectrum.

```bash
python train_forward.py \
    --train_path ./data/train.arrow \
    --dev_path   ./data/dev.arrow \
    --d_model 512 --n_layers 6 --n_heads 8 \
    --epochs 200 --batch_size 256 \
    --run_name forward_v1
```

### InverseModel

Loss: sum of two independent terms, both normalised by the number of non-padding tokens:

- **Material head**: label-smoothed KL divergence (`smoothing=0.1`). Prevents overconfidence on categorical material predictions.
- **Thickness head**: masked MSE in nm. Only non-padding positions contribute. No smoothing — the target is a single continuous value.

```bash
python train_inverse.py \
    --train_path ./data/train.arrow \
    --dev_path   ./data/dev.arrow \
    --d_model 512 --n_layers 6 --n_heads 8 \
    --epochs 200 --batch_size 256 \
    --run_name inverse_v1
```

Both scripts save `*_best.pt` (lowest dev loss) and `*_latest.pt` each epoch under `saved_models/forward/` or `saved_models/inverse/`.

Each checkpoint contains: `model` state dict, `optimizer` state, `epoch`, `loss_history`, `config`, and `vocab` (material word2id mapping).

---

## Evaluation

### ForwardModel evaluation

```bash
python evaluate.py \
    --checkpoint saved_models/forward/forward_best.pt \
    --val_path ./data/val.arrow \
    --plot_dir ./plots/forward_eval
```

Runs the model on the validation set and reports:

| Metric | Description |
|---|---|
| MSE | Mean squared error across all wavelength bins |
| MAE | Mean absolute error across all wavelength bins |
| R² | Coefficient of determination (global, across all samples) |

Plots saved to `--plot_dir`:
- `loss_curve.png` — train/dev loss curves (log scale)
- `sample_*.png` — 10 randomly sampled predicted vs. ground-truth spectra
- `scatter.png` — predicted vs target scatter plot
- `metrics.json` — numeric results

### InverseModel evaluation

```bash
python evaluate.py \
    --checkpoint saved_models/inverse/inverse_best.pt \
    --val_path ./data/val.arrow \
    --nk_dir ./nk \
    --n_samples 1000 \
    --plot_dir ./plots/inverse_eval
```

Evaluation proceeds in two stages:

1. **Greedy decoding**: for each target spectrum, autoregressively decode a predicted thin-film structure. The material decoder takes the argmax of its log-probability distribution; the thickness decoder takes the raw float output directly.

2. **TMM simulation**: simulate the predicted structure to obtain a spectrum, then compare against the original target. This is the physically meaningful metric — a structurally different prediction can still match the target spectrum.

Reported metrics are the same spectral MSE/MAE/R² as the forward model, computed between the simulated predicted spectrum and the input target. In addition to spectrum plots, `design_*.png` files show target vs predicted film stacks (materials and thicknesses) alongside their spectra.

### Comparison with optogpt v1

```bash
python compare.py \
    --ckpt_v1  ../optogpt/saved_models/ol_transformer.pt \
    --ckpt_v2  ./saved_models/forward/forward_best.pt \
    --plot_dir ./plots/comparison
```

Runs both models on the same validation samples and produces:
- Side-by-side spectrum comparison plots
- Metric bar chart (MSE, MAE, R²)
- Overlaid training loss curves
- `metrics.json` with numeric results for both models

Gracefully degrades to v2-only output if the v1 codebase is not available.

---

## Project structure

```
optogptv2/
├── optogptv2/
│   ├── data/
│   │   ├── sim.py          # TMM simulation, material loading
│   │   └── dataset.py      # Vocab, Batch, PrepareData
│   ├── model/
│   │   └── transformer.py  # All model components + factory functions
│   ├── training/
│   │   └── train.py        # CosineAnnealing+Warmup, LabelSmoothing, train_forward, train_inverse
│   └── eval/
│       ├── metrics.py      # SpectrumMetrics, forward_metrics, inverse_metrics
│       ├── decode.py       # Greedy decoding for InverseModel
│       └── visualize.py    # Matplotlib figure helpers
├── generate_data.py        # CLI: sample + simulate training data
├── train_forward.py        # CLI: train ForwardModel
├── train_inverse.py        # CLI: train InverseModel
├── evaluate.py             # CLI: evaluate a checkpoint
├── compare.py              # CLI: compare v1 vs v2
├── nk/                     # Per-material n,k CSV files
├── data/                   # Generated train.arrow / dev.arrow / val.arrow
└── pyproject.toml
```

---

## Planned experiments

### Architecture A vs B benchmark

Train both architectures on the same dataset and compare on the dev set.

| Metric | ForwardModel (MSE, MAE, R²) | InverseModel (spectral MSE after TMM simulation) |
|---|---|---|
| Architecture A | — | — |
| Architecture B (per-layer nm) | — | — |
| Architecture B (cumulative nm) | — | — |

**Variants of Architecture B to test:**
- `positions = thk_val[i]` — raw nm per layer
- `positions = cumsum(thk_vals)` — cumulative optical depth in nm; gives each layer a unique position even if two adjacent layers share the same thickness

**Controls to hold fixed across runs:**
- Same dataset, random seed, and train/dev split
- Same hyperparameters (`d_model`, `n_layers`, `n_heads`, `d_ff`, `dropout`)
- Same optimiser and warmup schedule

**Questions to answer:**
1. Does removing thickness from the embedding space hurt the ForwardModel when thickness is instead encoded via RoPE?
2. Does per-layer or cumulative thickness produce better RoPE positions?
3. Is there a measurable difference in the InverseModel's ability to predict physically accurate thicknesses?

---

## Differences from optogpt

| | optogpt | optogptv2 (Arch A) | optogptv2 (Arch B) |
|---|---|---|---|
| Token representation | Fused `Ta2O5_300` token | Separate material + thickness | Separate material + thickness |
| Thickness in embedding | Yes (categorical lookup) | Yes (linear projection of nm float) | No — thickness used as RoPE position only |
| Thickness as RoPE position | No | No | Yes (per-layer or cumulative nm) |
| Thickness representation | Categorical (50 discrete values) | Continuous float, regression | Continuous float, regression |
| Material vocab size | ~900 fused tokens | ~20 material tokens | ~20 material tokens |
| Positional encoding | Sinusoidal additive | RoPE, sequential positions | RoPE, nm thickness positions |
| InverseModel decoder | Single decoder + two heads | Single decoder + two heads | Single decoder + two heads |
| Thickness loss | Label-smoothed KL | Masked MSE (nm) | Masked MSE (nm) |
| FFN activation | Missing (linear only) | GELU | GELU |
| Data format | Pickle | Arrow (Feather) | Arrow (Feather) |
| Environment | conda | pip / uv | pip / uv |
