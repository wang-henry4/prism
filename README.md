# optoformer

A transformer-based model for **inverse** thin-film optical design: given a target optical spectrum, autoregressively generate a thin-film stack (materials + thicknesses) that produces it.

Built on top of [optogpt](https://github.com/taigaoma1997/optogpt). The key architectural change is a **dual-sequence representation**: materials and thicknesses are separated into two independent streams rather than fused into a single token. The thickness stream is treated as a **continuous float** (nm), not a categorical token.

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
python generate_data.py --n_samples 3000000
```

Outputs partitioned Arrow files into `data/train/`, `data/dev/`, `data/val/`. Running again auto-increments the partition number (e.g. `part_001.arrow`). Each run uses a random seed by default; pass `--seed 42` for reproducibility.

### 3. Train

```bash
python train_inverse.py \
    --train_path ./data/train --dev_path ./data/dev \
    --d_model 256 --n_layers 4 --n_heads 4 \
    --epochs 30 --batch_size 1024 --run_name inverse_v1
```

Checkpoints are saved to `saved_models/inverse/{run_name}/` as `best.pt` (lowest dev loss) and `latest.pt` (every epoch).

### 4. Evaluate

```bash
python evaluate.py \
    --checkpoint saved_models/inverse/inverse_v1/best.pt \
    --val_path ./data/val/part_000.arrow \
    --nk_dir ./nk --n_samples 1000 \
    --beam_width 5 --length_penalty 0.3 \
    --plot_dir ./plots/inverse_eval
```

Evaluation decodes structures (greedy or beam search), re-simulates them via TMM, and compares against target spectra. Also evaluates hand-crafted target spectra with top-K beam candidates. Writes `metrics.json`, scatter plot, loss curves, gradient stats, design comparisons, and beam candidate plots.

---

## Design Space

| Parameter | Range |
|---|---|
| Materials | 17 options (Al, Al2O3, AlN, Ge, HfO2, ITO, MgF2, MgO, Si, Si3N4, SiO2, Ta2O5, TiN, TiO2, ZnO, ZnS, ZnSe) + Glass_Substrate |
| Thicknesses | 5-250 nm, 5 nm steps |
| Layers per stack | 1-10 |
| Wavelength range | 400-1100 nm, 10 nm steps (71 wavelengths) |
| Spectrum | 142 floats (71 reflectance + 71 transmittance) |

---

## Model Architecture: Augmented Regression RoPE Thickness Encoding

The active model (`prefix_material_thk_model.py`) uses:

- **Spectrum prefix**: target spectrum projected to a single `[B, 1, d_model]` token, prepended to the decoder sequence. The decoder uses causal **self-attention** only (no cross-attention).
- **Material embedding**: standard learned embedding (no thickness fusion).
- **RoPE with cumulative depth**: thickness values are used directly as RoPE positions via `cumsum(thk_vals)`, encoding physical depth in nm rather than sequential position.
- **Per-material thickness head**: a multi-layer MLP that outputs `[B, T, vocab_size]` — one thickness prediction per material. This enables beam search to jointly score (material, thickness) pairs without committing to a material first.
- **Material head**: linear projection to `[B, T, vocab_size]` logits.

```
spectrum [B, 142] -> SpectrumProjection -> [SPEC] prefix token
                                                |
mat_ids  [B, T] -> MaterialEmbedding -> [tok1, tok2, ...]
                                                |
                         concat: [SPEC, tok1, tok2, ...] — causal self-attention
                         RoPE positions: [0, cumsum(thk)]
                                                |
                                         Encoder layers × N
                                                |
                                    hidden [B, T, d_model]  (prefix stripped)
                                                |
                                   +------------+------------+
                                   |                         |
                               mat_head                  thk_head (MLP)
                         log P(mat) [B, T, V]      thk_preds [B, T, V]
```

### Other model variants (archived)

These earlier designs are preserved in the codebase but are not used by the training or evaluation scripts:

| Variant | Module | Description |
|---|---|---|
| Thickness Embedding | `thickness_embedding_model.py` | Thickness projected as learned linear embedding added to material embedding; RoPE uses sequential positions |
| RoPE Thickness Encoding | `thickness_rope_model.py` | Thickness used directly as RoPE positions (cumulative depth); cross-attention to spectrum memory |
| Prefix RoPE Thickness Encoding | `prefix_model.py` | Same as RoPE Thickness Encoding but spectrum is a prefix token with causal self-attention; single scalar thickness head |

### Shared building blocks (`common.py`)

| Component | Detail |
|---|---|
| `MultiHeadAttention` | `w_q/w_k/w_v/w_o` projections; RoPE on Q and K; accepts optional `positions` tensor |
| `FeedForward` | `Linear -> GELU -> Dropout -> Linear` |
| `ResidualConnection` | Pre-norm: `x + dropout(sublayer(LayerNorm(x)))` |

Weight initialisation: Xavier uniform for all weight matrices (dim > 1).

---

## Training

### Loss

Sum of two terms, normalised by non-padding token count:
- **Material head**: label-smoothed KL divergence (`smoothing=0.1`)
- **Thickness head**: masked MSE (nm), scaled by `--thk_loss_weight` (default `0.001`) to balance against material KL

Per-token component losses (`mat`, `thk`) and their ratio are logged each epoch to help tune the weight.

### Learning rate schedule

Cosine annealing with linear warmup: `peak_lr=3e-4`, `warmup_steps=4000`, `min_lr=3e-7`.

### Gradient monitoring

Both grad L2 norm and max absolute gradient are logged per step and saved as epoch-level summaries (mean/max) in the checkpoint's `loss_history`.

---

## Evaluation

### Decoding strategies

| Strategy | Flag | Description |
|---|---|---|
| Greedy | `--beam_width 1` | Argmax at each step |
| Beam search | `--beam_width 5` (default) | Length-normalised beam search with configurable `--length_penalty` (default 0.3) |

The per-material thickness head enables beam search to score each candidate material with its own thickness prediction, rather than using a single shared thickness.

### Hand-crafted targets

A set of hand-crafted target spectra (e.g. shortpass filters) are always evaluated during `evaluate.py`. For each target, the top-K beam candidates are decoded, TMM re-simulated, and plotted showing all candidates' film stacks and spectra.

### Pipeline

1. **Decode**: generate predicted structures from target spectra
2. **TMM re-simulate**: simulate predicted structures to obtain spectra
3. **Compare**: spectral MSE, MAE, R^2 between simulated and target spectra

### Outputs

- `metrics.json` - MSE, MAE, R^2
- `scatter.png` - predicted vs target scatter plot
- `loss_curve.png` - train/dev loss curves
- `loss_components.png` - per-token mat vs thk loss and their ratio
- `grad_stats.png` - gradient norm/max over training
- `design_*.png` - film stack + spectrum comparisons for random samples
- `handcrafted/*.png` - top-K beam candidates for hand-crafted targets

---

## Data

### Format

Arrow (Feather) files, partitioned by split:

```
data/
  train/
    part_000.arrow
    part_001.arrow
    ...
  dev/
    part_000.arrow
  val/
    part_000.arrow
```

The dataloader accepts a directory path and loads all `part_*.arrow` files via `ConcatDataset`. Data is lazy-loaded via memory-mapped Arrow tables.

### Generation

```bash
# First run (random seed by default)
python generate_data.py --n_samples 3000000

# Add more data (auto-increments partition number)
python generate_data.py --n_samples 1000000

# Reproducible run
python generate_data.py --n_samples 3000000 --seed 42
```

---

## Project Structure

```
optoformer/
  optoformer/
    data/
      sim.py              # TMM simulation, material loading
      dataset.py           # Vocab, Batch, ThinFilmDataset, make_dataloader
    model/
      common.py            # Shared building blocks (attention, FFN, RoPE)
      prefix_material_thk_model.py  # Augmented Regression RoPE Thickness Encoding (active)
      prefix_model.py               # Prefix RoPE Thickness Encoding (archived)
      thickness_rope_model.py       # RoPE Thickness Encoding (archived)
      thickness_embedding_model.py  # Thickness Embedding (archived)
      transformer.py       # Re-exports InverseModel from active architecture
    training/
      train.py             # LR schedule, LabelSmoothing, train_inverse
    eval/
      metrics.py           # SpectrumMetrics (MSE, MAE, R^2)
      decode.py            # Greedy + beam search decoding (incl. top-K)
      visualize.py         # Matplotlib figure helpers
      targets.py           # Hand-crafted target spectra registry
  generate_data.py         # CLI: sample + simulate training data
  train_inverse.py         # CLI: train InverseModel
  evaluate.py              # CLI: evaluate a checkpoint
  nk/                      # Per-material n,k CSV files
  data/                    # Generated partitioned Arrow files
  pyproject.toml
```
