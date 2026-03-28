# Baselines

Baseline evaluations on the optoformer val set (`data/max_len_20/val`, 10,000 samples) and 29 handcrafted practical filter targets.

## Evaluation Protocol

1. Model receives a target spectrum (142 floats: 71 R + 71 T, 400–1100 nm)
2. Model autoregressively predicts a thin-film structure (materials + thicknesses)
3. Predicted structure is re-simulated via TMM (transfer matrix method)
4. Re-simulated spectrum is compared against the target

Structures are **not** compared directly — only spectral fidelity matters, since inverse design is one-to-many.

## Val Set Results

| Model | Decoding | MAE ↓ | MSE ↓ | R² ↑ | Median MAE | P90 MAE |
|-------|----------|-------|-------|------|------------|---------|
| OptoGPT | greedy | 0.0672 | 0.0262 | 0.681 | 0.0364 | 0.1965 |

## Handcrafted Target Results

29 practical optical filter spectra (narrowband, broadband, edge, notch, dichroic, neutral density, multi-bandpass, hot/cold mirror, linear variable).

| Model | Decoding | Mean MAE ↓ | Median MAE ↓ |
|-------|----------|------------|--------------|
| OptoGPT | greedy | 0.3499 | 0.3419 |

### OptoGPT per-category breakdown

| Category | # Targets | Mean MAE |
|----------|-----------|----------|
| Notch filters | 4 | 0.103 |
| Edge filters | 6 | 0.326 |
| Dichroic | 3 | 0.329 |
| Hot/cold mirror | 2 | 0.329 |
| Broadband bandpass | 3 | 0.435 |
| Neutral density | 5 | 0.408 |
| Multi-bandpass | 2 | 0.463 |
| Narrowband | 3 | 0.500 |

OptoGPT only performs well on notch filters (mostly-flat spectra with a narrow dip). It fails on targets requiring precise spectral shaping (narrowband, ND, multi-bandpass all hit MAE ≈ 0.50).

## OptoGPT

Pretrained checkpoint from [huggingface.co/mataigao/optogpt](https://huggingface.co/mataigao/optogpt) (`optogpt.pt`, epoch 146).

Architecture: 6-layer decoder-only transformer, d_model=1024, 8 heads, d_ff=512. Vocab of 904 discrete `Material_Thickness` tokens (18 materials × 50 thickness buckets at 10 nm steps, 10–500 nm). Max sequence length 22 (up to 20 layers).

**Distribution mismatch:** The val set uses 5 nm thickness resolution (5–250 nm range) while OptoGPT was trained on 10 nm resolution (10–500 nm range).

## Reproduce

```bash
python baselines/eval_optogpt.py \
    --checkpoint /path/to/optogpt.pt \
    --val_path ./data/max_len_20/val/part_000.arrow \
    --nk_dir ./nk
```

Plots saved to `plots/baselines/optogpt/`, including per-sample design comparisons, scatter plot, and per-target handcrafted results.
