# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PRISM** (Position-encoded Regressive Inverse Spectral Model) is a transformer-based model for thin-film optical design. The core innovations are a **dual-head, single-backbone architecture** (shared decoder with separate material and thickness output heads) and **cumulative-depth RoPE positional encoding**, with thickness treated as a continuous float (nm) instead of a categorical token.

The architecture (`prefix_material_thk_model.py`):
- Spectrum projected as a prefix token; causal self-attention only (no cross-attention)
- Material-only learned embedding; thickness encoded via cumulative-depth RoPE
- Per-material thickness head: multi-layer MLP with softplus activation (log-space), outputting `[B, T, vocab_size]` — one thickness prediction per material, enabling joint (material, thickness) beam search

## Setup

```bash
uv sync
# or
pip install -e .
```

Requires Python 3.10+. The `nk/` directory contains 42 per-material CSV files with columns `wl` (µm), `n`, `k` and must be present for data generation and inverse evaluation.

## Workflow Commands

### Data generation
```bash
python generate_data.py --n_samples 3000000
```
Outputs partitioned Arrow files into `data/train/`, `data/dev/`, `data/val/`. Running again auto-increments the partition number.

### Training
```bash
python train_inverse.py \
    --train_path ./data/train --dev_path ./data/dev \
    --d_model 256 --n_layers 4 --n_heads 4 \
    --epochs 30 --batch_size 1024 --run_name inverse_v1
```

Checkpoints saved to `saved_models/inverse/{run_name}/` as `best.pt` (lowest dev loss) and `latest.pt` (each epoch). Each checkpoint contains: model state dict, optimizer state, epoch, loss_history, config, vocab.

### Evaluation
```bash
python evaluate.py \
    --checkpoint saved_models/inverse/inverse_v1/best.pt \
    --val_path ./data/val/part_000.arrow \
    --nk_dir ./nk --n_samples 1000 \
    --beam_width 5 --length_penalty 0.3 \
    --plot_dir ./plots/inverse_eval
```

## Architecture

### Package structure
```
prism/
├── data/
│   ├── sim.py        # TMM simulation, nk CSV loading, cubic-spline interpolation
│   └── dataset.py    # Vocab, Batch, ThinFilmDataset, make_dataloader
├── model/
│   ├── common.py                    # Shared building blocks (attention, FFN, RoPE)
│   ├── prefix_material_thk_model.py # InverseModel: spectrum prefix + RoPE + dual heads
│   └── transformer.py               # Re-exports InverseModel
├── training/
│   └── train.py      # CosineAnnealing+Warmup, LabelSmoothing, train_inverse
└── eval/
    ├── metrics.py    # SpectrumMetrics (MSE, MAE, R²)
    ├── decode.py     # Greedy + beam search decoding (incl. top-K)
    ├── visualize.py  # Matplotlib figure helpers
    └── targets.py    # Hand-crafted target spectra registry
```

### Key model components (`prefix_material_thk_model.py`)

- **MaterialEmbedding**: `Embedding(mat_id) × √d_model` — thickness not fused here
- **SpectrumProjection**: maps 142-float spectrum → `[B, 1, d_model]` prefix token
- **ThicknessMLPHead**: multi-layer MLP → `[B, T, vocab_size]` per-material thickness predictions
- **InverseModel**: spectrum prefix + causal self-attention + cumulative-depth RoPE + dual output heads

### Shared building blocks (`common.py`)

- **MultiHeadAttention**: `w_q/w_k/w_v/w_o` projections; RoPE on Q and K; accepts optional `positions` tensor
- **FeedForward**: `Linear → GELU → Dropout → Linear`
- **ResidualConnection**: pre-norm — `x + dropout(sublayer(LayerNorm(x)))`

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

`d_model=256`, `d_ff=1024`, `n_heads=4`, `n_layers=4`, `dropout=0.1`, `thk_head_hidden_layers=2`

### Learning rate schedule

Cosine annealing with linear warmup: `peak_lr=3e-4`, `warmup_steps=4000`, `min_lr=3e-7`

### Loss functions

- **Material head**: label-smoothed KL divergence (`smoothing=0.1`)
- **Thickness head**: masked MSE in log-space, scaled by `thk_loss_weight` (default `1.0`)
- Both normalised by non-padding token count. Per-token component losses logged each epoch.

### RoPE positions

`positions = cumsum(thk_vals)` — cumulative depth in nm
