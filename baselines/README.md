# Baselines

Baseline evaluations on the `max_len_20_10nm` val set (10,000 samples, 10 nm thickness steps, 10–500 nm, up to 20 layers) and 29 handcrafted practical filter targets.

## Evaluation Protocol

1. Model receives a target spectrum (142 floats: 71 R + 71 T, 400–1100 nm)
2. Model autoregressively predicts a thin-film structure (materials + thicknesses)
3. Predicted structure is re-simulated via TMM (transfer matrix method)
4. Re-simulated spectrum is compared against the target

Structures are **not** compared directly — only spectral fidelity matters, since inverse design is one-to-many.

## Val Set Results

| Model | Decoding | MAE ↓ | MSE ↓ | R² ↑ |
|-------|----------|-------|-------|------|
| OptoGPT | greedy | 0.0585 | 0.0218 | 0.715 |

## Handcrafted Target Results

29 practical optical filter spectra (narrowband, broadband, edge, notch, dichroic, neutral density, multi-bandpass, hot/cold mirror, linear variable).

| Model | Decoding | Mean MAE ↓ | Mean MSE ↓ | Mean R² ↑ |
|-------|----------|------------|------------|-----------|
| OptoGPT | greedy | 0.3499 | 0.2332 | -5.40 |

### OptoGPT per-category breakdown

| Category | # Targets | Mean MAE | Mean MSE | Mean R² |
|----------|-----------|----------|----------|---------|
| Notch filters | 4 | 0.103 | 0.019 | 0.933 |
| Edge filters | 6 | 0.326 | 0.157 | 0.348 |
| Dichroic | 3 | 0.329 | 0.168 | 0.111 |
| Hot/cold mirror | 2 | 0.329 | 0.155 | 0.307 |
| Broadband bandpass | 3 | 0.435 | 0.253 | -0.697 |
| Neutral density | 5 | 0.408 | 0.218 | -31.47 |
| Multi-bandpass | 2 | 0.463 | 0.260 | -0.515 |
| Narrowband | 3 | 0.500 | 0.250 | -1.024 |
| Linear variable | 1 | 0.364 | 0.157 | 0.187 |

OptoGPT only performs well on notch filters (mostly-flat spectra with a narrow dip). It fails on targets requiring precise spectral shaping (narrowband, ND, multi-bandpass all hit MAE ≈ 0.50).

## OptoGPT

Pretrained checkpoint from [huggingface.co/mataigao/optogpt](https://huggingface.co/mataigao/optogpt) (`optogpt.pt`, epoch 146).

Architecture: 6-layer decoder-only transformer, d_model=1024, 8 heads, d_ff=512. Vocab of 904 discrete `Material_Thickness` tokens (18 materials × 50 thickness buckets at 10 nm steps, 10–500 nm). Max sequence length 22 (up to 20 layers).

## Reproduce

```bash
python baselines/eval_optogpt.py \
    --checkpoint /path/to/optogpt.pt \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow \
    --nk_dir ./nk \
    --plot_dir ./plots/baselines/optogpt_10nm
```

Plots saved to `plots/baselines/optogpt_10nm/`, including per-sample design comparisons, scatter plot, and per-target handcrafted results in `handcrafted/`.
