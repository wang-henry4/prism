# Baselines

Baseline evaluations on the `max_len_20_10nm` val set and 29 handcrafted practical filter targets.

## Evaluation Protocol

1. Model/optimizer receives a target spectrum (142 floats: 71 R + 71 T, 400–1100 nm)
2. Produces a thin-film structure (materials + thicknesses)
3. Predicted structure is re-simulated via TMM (transfer matrix method)
4. Re-simulated spectrum is compared against the target

Structures are **not** compared directly — only spectral fidelity matters, since inverse design is one-to-many.

## Val Set Results

| Method | Type | Val Samples | MAE ↓ | MSE ↓ | R² ↑ |
|--------|------|-------------|-------|-------|------|
| OptoGPT | Neural (autoregressive) | 10,000 | 0.0585 | 0.0218 | 0.715 |
| Differentiable TMM | Gradient optimization | 1,000 | 0.0306 | 0.0037 | 0.954 |
| Simulated Annealing | Global stochastic search | 1,000 | 0.0163 | 0.0017 | 0.978 |

## Handcrafted Target Results

29 practical optical filter spectra (narrowband, broadband, edge, notch, dichroic, neutral density, multi-bandpass, hot/cold mirror, linear variable).

| Method | Type | Mean MAE ↓ | Mean MSE ↓ | Mean R² ↑ |
|--------|------|------------|------------|-----------|
| OptoGPT | Neural (autoregressive) | 0.3499 | 0.2332 | -5.40 |
| Differentiable TMM | Gradient optimization | 0.1864 | 0.0863 | 0.573 |
| Simulated Annealing | Global stochastic search | 0.1216 | 0.0450 | 0.788 |

## Methods

### OptoGPT (pretrained baseline)
Pretrained checkpoint from [huggingface.co/mataigao/optogpt](https://huggingface.co/mataigao/optogpt) (`optogpt.pt`, epoch 146). 6-layer decoder-only transformer, d_model=1024, 8 heads, d_ff=512. Vocab of 904 discrete `Material_Thickness` tokens (18 materials × 50 thickness buckets at 10 nm steps). Greedy decoding.

### Differentiable TMM (gradient-based inverse design)
PyTorch-differentiable TMM implementation enabling direct gradient-based optimization. For each target spectrum: pick random materials and layer count, optimize thicknesses with L-BFGS. 32 random restarts across layer counts [3, 5, 7, 10, 14, 18], 300 L-BFGS iterations per start. No training required.

### Simulated Annealing (global stochastic search)
Classical SA over the joint (materials, thicknesses, layer count) space. Moves: thickness perturbation (50%), material swap (20%), layer insertion (15%), layer removal (15%). Exponential temperature schedule from 0.1 to 1e-4 over 5000 steps, 8 random restarts. No training required.

## Runtime

Optimization methods are per-sample (no amortization):

| Method | Val (per sample) | Handcrafted (per target) |
|--------|-----------------|--------------------------|
| OptoGPT | ~0.04s | ~0.04s |
| Differentiable TMM | ~50s | ~50s |
| Simulated Annealing | ~160s | ~160s |

Neural methods (OptoGPT) are orders of magnitude faster at inference, but optimization methods achieve better spectral fidelity since they directly optimize against the physics.

## Reproduce

```bash
# OptoGPT
python baselines/eval_optogpt.py \
    --checkpoint /path/to/optogpt.pt \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow \
    --nk_dir ./nk --plot_dir ./plots/baselines/optogpt_10nm

# Differentiable TMM
python baselines/eval_optim.py --method diff_tmm \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow \
    --nk_dir ./nk --n_val_samples 1000

# Simulated Annealing
python baselines/eval_optim.py --method sa \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow \
    --nk_dir ./nk --n_val_samples 1000
```
