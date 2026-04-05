# Optoformer 13M Model Evaluation Report

## 1. Model Overview

| Property | Value |
|---|---|
| **Model** | Optoformer (Augmented Regression RoPE Thickness Encoding) |
| **Parameters** | ~13M |
| **Architecture** | `d_model=256`, `n_layers=4`, `n_heads=4`, `d_ff=1024` |
| **Thickness head** | 2-hidden-layer MLP, log-space output |
| **Checkpoint** | `saved_models/inverse/optoformer_len_20_13m_10nm/best.pt` (epoch 29) |

### Training Data

- **Sequence length**: 1--20 layers
- **Thickness step**: 10 nm increments
- **Thickness range per layer**: 10--500 nm
- **Maximum cumulative depth**: 20 x 500 = 10,000 nm
- **Materials**: 17 dielectrics/metals + Glass substrate
- **Spectrum**: 142 values (71 reflectance + 71 transmittance, 400--1100 nm at 10 nm spacing)

### Training Configuration

- **Loss**: Label-smoothed KL divergence (material head) + masked MSE in nm (thickness head, weight 0.001)
- **Learning rate**: Cosine annealing with linear warmup (peak 3e-4, 4000 warmup steps, min 3e-7)
- **Dropout**: 0.1

---

## 2. Evaluation Methodology

### Decoding

All evaluations use autoregressive decoding with two strategies:

1. **Greedy decode**: Selects the highest-probability material at each step; thickness predicted by the per-material MLP head.
2. **Beam search** (width 5, length penalty 0.3): Explores multiple candidate designs per input spectrum. Reports both:
   - **Top-1**: The highest-scoring beam candidate
   - **Oracle-best**: The beam candidate with the lowest MSE against the ground truth (upper bound on beam re-ranking)

### Re-simulation

Decoded designs (material sequences + thickness sequences) are re-simulated using the Transfer Matrix Method (TMM) with cubic-spline-interpolated nk data. This ensures metrics reflect physically accurate spectra, not raw model logits.

### Metrics

All metrics are computed over the full 142-dimensional spectrum vector (71 R + 71 T):

| Metric | Definition |
|---|---|
| **MSE** | Mean Squared Error: `mean((pred - target)^2)` |
| **MAE** | Mean Absolute Error: `mean(|pred - target|)` |
| **R^2** | Coefficient of determination: `1 - SS_res / SS_tot` |

### Handcrafted Targets

In addition to validation-set evaluation, each configuration is tested against a suite of ~25 handcrafted target spectra (bandpass, longpass, shortpass, notch, neutral density, dichroic, etc.) decoded with beam search (width >= 5) and TMM-resimulated.

### Sample Size

All evaluations use **n=10,000** validation samples.

---

## 3. In-Distribution Results

The model is evaluated on data drawn from the same distribution as training (10 nm thickness steps, 1--20 layers).

### 3.1 In-Distribution: 10 nm Steps

**Data**: `data/max_len_20_10nm/val/part_000.arrow`

| Decoding | MSE | MAE | R^2 |
|---|---|---|---|
| Greedy | 0.00837 | 0.0379 | 0.890 |
| Beam Top-1 | 0.02584 | 0.0646 | 0.662 |
| **Oracle-best** | **0.00656** | **0.0326** | **0.914** |

The model achieves strong in-distribution performance with R^2 = 0.914 (oracle) and R^2 = 0.890 (greedy). Greedy decoding outperforms beam top-1, indicating that beam search score ranking does not always correlate with spectral fidelity; however, the oracle metric shows that beam search does surface better candidates -- the gap between greedy and oracle represents the headroom available through improved re-ranking.

---

## 4. Out-of-Distribution Results

The model was tested on data with thickness step sizes it was **not trained on** (5 nm, 15 nm, 20 nm). All OOD datasets use 20-layer designs. Since the model uses cumulative-depth RoPE positions (not integer layer indices), OOD thickness steps produce position values outside the training range, making this a test of both thickness granularity and positional generalization.

### 4.1 OOD: 5 nm Thickness Steps

**Data**: `data/max_len_20/val/part_000.arrow` (5 nm steps, max cumulative depth ~2,500 nm -- well within training range)

| Decoding | MSE | MAE | R^2 |
|---|---|---|---|
| Greedy | 0.00510 | 0.0320 | 0.938 |
| Beam Top-1 | 0.01689 | 0.0490 | 0.794 |
| **Oracle-best** | **0.00395** | **0.0278** | **0.952** |

The model performs **better** on 5 nm data than on in-distribution 10 nm data. This is expected: 5 nm steps with 20 layers produce a maximum cumulative depth of ~2,500 nm, well within the training range of 10,000 nm. The finer granularity also means simpler optical structures that are easier to reconstruct.

### 4.2 OOD: 15 nm Thickness Steps

**Data**: `data/max_len_20_15nm/val/part_000.arrow` (15 nm steps, max cumulative depth ~7,500 nm)

Several RoPE context extension methods were tested alongside a no-scaling baseline:

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.01297** | **0.828** | **0.01004** | **0.867** |
| **Dynamic NTK** | 1.5 | **0.01297** | **0.828** | **0.01004** | **0.867** |
| PI (Position Interpolation) | 1.5 | 0.04171 | 0.448 | 0.03609 | 0.523 |
| NTK-aware | 1.5 | 0.03142 | 0.584 | 0.02783 | 0.632 |
| YaRN | 1.5 | 0.03178 | 0.580 | 0.02769 | 0.634 |

**Key findings:**
- No scaling and Dynamic NTK produce identical results (expected, since 15 nm x 20 layers = 7,500 nm < 10,000 nm training max, so Dynamic NTK does not activate).
- All static scaling methods (PI, NTK, YaRN) **degrade** performance, because the cumulative depths are still within the training range -- scaling distorts positions unnecessarily.

#### 15 nm -- Thick Designs Only (cumulative depth >= 11,000 nm)

**Data**: `data/thick/15nm/val/part_000.arrow`, filtered to cumulative depth >= 11,000 nm

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.02271** | **0.661** | **0.01906** | **0.715** |
| **Dynamic NTK** | 1.5 | **0.02271** | **0.661** | **0.01906** | **0.715** |
| NTK-aware | 1.5 | 0.04418 | 0.340 | 0.03868 | 0.422 |
| YaRN | 1.5 | 0.04511 | 0.326 | 0.03823 | 0.429 |

Even for thick designs that exceed the training cumulative depth range, no scaling and Dynamic NTK remain the best approaches. The static methods uniformly hurt.

### 4.3 OOD: 20 nm Thickness Steps

**Data**: `data/max_len_20_20nm/val/part_000.arrow` (20 nm steps, max cumulative depth ~10,000 nm)

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.01798** | **0.750** | **0.01536** | **0.787** |
| **Dynamic NTK** | 2.0 | **0.01798** | **0.750** | **0.01536** | **0.787** |
| NTK-aware | 2.0 | 0.04851 | 0.326 | 0.04239 | 0.411 |
| PI | 2.0 | 0.07491 | -0.040 | 0.06760 | 0.061 |
| YaRN | 2.0 | 0.04896 | 0.320 | 0.04297 | 0.403 |

**Key findings:**
- No scaling and Dynamic NTK again tie and lead. The 20 nm x 20 layers = 10,000 nm maximum cumulative depth matches the training range exactly, so Dynamic NTK does not trigger.
- PI at scale factor 2.0 catastrophically fails (R^2 near zero), compressing all positions into half the training range.
- NTK and YaRN perform similarly to each other but are substantially worse than no scaling.

#### 20 nm -- Thick Designs Only (cumulative depth >= 11,000 nm)

**Data**: `data/thick/20nm/val/part_000.arrow`, filtered to cumulative depth >= 11,000 nm

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.02735** | **0.589** | **0.02351** | **0.646** |
| **Dynamic NTK** | 2.0 | **0.02735** | **0.589** | **0.02351** | **0.646** |
| NTK-aware | 2.0 | 0.05977 | 0.101 | 0.05388 | 0.189 |
| YaRN | 2.0 | 0.06143 | 0.076 | 0.05423 | 0.184 |

Same pattern holds. Even when cumulative depth exceeds the training maximum (11,000+ nm), the unscaled baseline outperforms all RoPE extension methods. Dynamic NTK matches the baseline because its scaling only activates proportionally when the max position exceeds the training range, which has a very modest effect at these depths.

---

## 5. Summary of Results

### Oracle R^2 Across All Conditions

| Eval Condition | Thickness Step | Cum. Depth Range | Oracle R^2 |
|---|---|---|---|
| 5 nm (OOD step) | 5 nm | ~250--2,500 nm | **0.952** |
| 10 nm (in-distribution) | 10 nm | ~100--5,000 nm | 0.914 |
| 15 nm (OOD step) | 15 nm | ~300--7,500 nm | 0.867 |
| 15 nm, thick (>= 11,000 nm) | 15 nm | >= 11,000 nm | 0.715 |
| 20 nm (OOD step) | 20 nm | ~400--10,000 nm | 0.787 |
| 20 nm, thick (>= 11,000 nm) | 20 nm | >= 11,000 nm | 0.646 |

### Key Takeaways

1. **Strong in-distribution performance**: The 13M model achieves oracle R^2 = 0.914 on 10 nm validation data, demonstrating effective inverse design of thin-film stacks up to 20 layers.

2. **Graceful OOD degradation with thickness step**: Performance degrades smoothly as the thickness step deviates from the training distribution (10 nm). Finer steps (5 nm) actually improve performance since they produce shallower, simpler structures. Coarser steps (15 nm, 20 nm) progressively reduce accuracy.

3. **Cumulative depth is the dominant OOD factor**: The largest performance drops occur when cumulative depth exceeds the training maximum of 10,000 nm (the "thick" evaluations at >= 11,000 nm). This confirms that the cumulative-depth RoPE encoding ties model capability directly to the position range seen during training.

4. **RoPE scaling methods do not help**: Across all OOD conditions tested, no RoPE context extension method (PI, NTK, Dynamic NTK, YaRN) improved over the unscaled baseline. Static methods actively harmed performance even on in-range data. Dynamic NTK was neutral (matched baseline) because the OOD depths were not extreme enough to trigger significant rescaling. This suggests the model's OOD limitations stem from the optical complexity of thicker designs, not from positional encoding limitations.

5. **Greedy vs. beam search**: Greedy decoding consistently outperforms beam top-1 scoring, but oracle-best beam candidates are reliably better than greedy. This indicates beam search generates diverse, high-quality candidates -- improved re-ranking (e.g., via TMM-based scoring) could close the gap between top-1 and oracle performance.

---

## 6. RoPE Scaling Methods Tested

For reference, the RoPE context extension methods evaluated:

| Method | Mechanism | Scale Factors Tested |
|---|---|---|
| **None** | Baseline -- no position scaling | -- |
| **PI** (Position Interpolation) | Divides all positions by scale factor | 1.5, 2.0 |
| **NTK-aware** | Increases RoPE base frequency: `base * s^(d/(d-2))` | 1.5, 2.0 |
| **Dynamic NTK** | Like NTK but only activates when max position exceeds training max (10,000 nm) | 1.5, 2.0 |
| **YaRN** | NTK base scaling + attention temperature correction (`1/sqrt(s)`) | 1.5, 2.0 |
