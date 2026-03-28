# Baselines

Baseline evaluations on the optoformer val set (`data/max_len_20/val/part_000.arrow`, 10,000 samples).

## Evaluation Protocol

1. Model receives a target spectrum (142 floats: 71 R + 71 T, 400–1100 nm)
2. Model autoregressively predicts a thin-film structure (materials + thicknesses)
3. Predicted structure is re-simulated via TMM (transfer matrix method)
4. Re-simulated spectrum is compared against the target

Structures are **not** compared directly — only spectral fidelity matters, since inverse design is one-to-many.

## Results

| Model | MAE ↓ | MSE ↓ | R² ↑ | Median MAE | P90 MAE |
|-------|-------|-------|------|------------|---------|
| OptoGPT | 0.0672 | 0.0262 | 0.681 | 0.0364 | 0.1965 |

## OptoGPT

Pretrained checkpoint from [huggingface.co/mataigao/optogpt](https://huggingface.co/mataigao/optogpt) (`optogpt.pt`, epoch 146).

Architecture: 6-layer decoder-only transformer, d_model=1024, 8 heads, d_ff=512. Vocab of 904 discrete `Material_Thickness` tokens (18 materials × 50 thickness buckets at 10 nm steps, 10–500 nm). Greedy decoding, max sequence length 22 (up to 20 layers).

**Note:** The val set uses 5 nm thickness resolution and up to 20 layers. OptoGPT was trained on 10 nm resolution, so there is a distribution mismatch on thickness granularity.

```bash
python baselines/eval_optogpt.py \
    --checkpoint /path/to/optogpt.pt \
    --val_path ./data/max_len_20/val/part_000.arrow \
    --nk_dir ./nk
```
