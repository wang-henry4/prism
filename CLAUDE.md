# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**optoformer** is a transformer-based model for thin-film optical design. The core innovation is a **dual-sequence representation** where materials and thicknesses are separate streams rather than fused tokens, with thickness treated as a continuous float (nm) instead of a categorical token.

Two model types:
- **ForwardModel** (encoder-only): thin-film structure → optical spectrum
- **InverseModel** (decoder + two output heads): target spectrum → thin-film structure (autoregressive)

Two architectural variants are being benchmarked:
- **Architecture A**: thickness projected as a learned linear embedding, added to material embedding; RoPE uses standard sequential positions
- **Architecture B**: thickness used directly as the RoPE position argument (cumulative depth in nm), no thickness embedding

## Setup

```bash
pip install -e .
# or
uv sync
```

Requires Python 3.10+. The `nk/` directory contains 42 per-material CSV files with columns `wl` (µm), `n`, `k` and must be present for data generation and inverse evaluation.

## Workflow Commands

### Data generation
```bash
python generate_data.py \
    --n_samples 100000 --min_layers 1 --max_layers 10 \
    --dev_split 0.1 --val_split 0.1 --out_dir ./data --nk_dir ./nk \
    --workers 8 --seed 42
```
Outputs `data/train.arrow`, `data/dev.arrow`, and `data/val.arrow`. Dev is used during training for early stopping; val is held out for final evaluation. Simulation is parallelised via `multiprocessing.Pool`.

### Training
```bash
# ForwardModel
python train_forward.py \
    --train_path ./data/train.arrow --dev_path ./data/dev.arrow \
    --d_model 512 --n_layers 6 --n_heads 8 \
    --epochs 200 --batch_size 256 --run_name forward_v1

# InverseModel
python train_inverse.py \
    --train_path ./data/train.arrow --dev_path ./data/dev.arrow \
    --d_model 512 --n_layers 6 --n_heads 8 \
    --epochs 200 --batch_size 256 --run_name inverse_v1
```

Checkpoints saved to `saved_models/forward/` and `saved_models/inverse/` as `*_best.pt` (lowest dev loss) and `*_latest.pt` (each epoch). Each checkpoint contains: model state dict, optimizer state, epoch, loss_history, config, vocab.

### Evaluation
```bash
# ForwardModel
python evaluate.py \
    --checkpoint saved_models/forward/forward_best.pt \
    --val_path ./data/val.arrow \
    --plot_dir ./plots/forward_eval

# InverseModel (with TMM re-simulation)
python evaluate.py \
    --checkpoint saved_models/inverse/inverse_best.pt \
    --val_path ./data/val.arrow \
    --nk_dir ./nk --n_samples 1000 \
    --plot_dir ./plots/inverse_eval

# Compare v1 vs v2
python compare.py \
    --ckpt_v1 ../optogpt/saved_models/ol_transformer.pt \
    --ckpt_v2 ./saved_models/forward/forward_best.pt \
    --val_path ./data/val.arrow \
    --plot_dir ./plots/comparison
```

## Architecture

### Package structure (to be implemented under `optoformer/`)
```
optoformer/
├── data/
│   ├── sim.py        # TMM simulation, nk CSV loading, cubic-spline interpolation
│   └── dataset.py    # Vocab, Batch, PrepareData
├── model/
│   └── transformer.py  # All model components + factory functions
├── training/
│   └── train.py      # CosineAnnealing+Warmup, LabelSmoothing, train_forward, train_inverse
└── eval/
    ├── metrics.py    # SpectrumMetrics (MSE, MAE, R²)
    ├── decode.py     # Greedy decoding for InverseModel
    └── visualize.py  # Matplotlib figure helpers
```

### Key model components (`optoformer/model/transformer.py`)

- **MultiHeadAttention**: projections named `w_q/w_k/w_v/w_o`; RoPE applied to Q and K only; accepts optional `positions` tensor (enables Arch B)
- **FeedForward**: `Linear → GELU → Dropout → Linear`
- **ResidualConnection**: pre-norm — `x + dropout(sublayer(LayerNorm(x)))`
- **DualEmbedding** (Arch A): `embed(i) = (Embedding_mat(mat_id[i]) + Linear_thk(thk_val[i] / 250)) × √d_model`
- **SpectrumHead**: maps CLS token hidden state → 142-float spectrum
- **SpectrumProjection**: maps 142-float spectrum → `memory [B, 1, d_model]` for cross-attention in InverseModel

Weight init: Xavier uniform for all weight matrices with dim > 1.

### Data format

```json
{
  "materials":   [["Ta2O5", "AlN", "SiO2"], ...],
  "thicknesses": [[300.0, 320.0, 150.0], ...],
  "spectra":     [[0.12, 0.14, ..., 0.33, 0.41, ...], ...]
}
```

Spectrum: 142 floats = 71 reflectance + 71 transmittance values at 400–1100 nm (10 nm steps).

### Design space

| Parameter | Value |
|---|---|
| Materials | 17 (Al, Al2O3, AlN, Ge, HfO2, ITO, MgF2, MgO, Si, Si3N4, SiO2, Ta2O5, TiN, TiO2, ZnO, ZnS, ZnSe) + Glass_Substrate |
| Thicknesses | 5–250 nm, 5 nm steps |
| Layers | 1–10 |
| Wavelengths | 71 points, 400–1100 nm |

### Default hyperparameters

`d_model=512`, `d_ff=2048`, `n_heads=8`, `n_layers=6`, `dropout=0.1`, `warmup_steps=4000`

### Learning rate schedule

Cosine annealing with linear warmup: `peak_lr=3e-4`, `warmup_steps=2000`, `min_lr=1e-6`

### Loss functions

- **InverseModel**: label-smoothed KL divergence (`smoothing=0.1`) for material head + masked MSE (nm) for thickness head; both normalised by non-padding token count

### Architecture B/C RoPE positions

`positions = cumsum(thk_vals)` — cumulative depth in nm
