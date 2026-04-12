# PRISM 13M Model Evaluation Report

## 1. Model Overview

| Property | Value |
|---|---|
| **Model** | PRISM (Position-encoded Regressive Inverse Spectral Model) |
| **Parameters** | ~13M |
| **Architecture** | `d_model=256`, `n_layers=4`, `n_heads=4`, `d_ff=1024` |
| **Thickness head** | 2-hidden-layer MLP, log-space output |
| **Checkpoint** | `saved_models/inverse/prism_len_20_13m_10nm/best.pt` (epoch 29) |

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

In addition to validation-set evaluation, each configuration is tested against a suite of 29 handcrafted target spectra (bandpass, longpass, shortpass, notch, neutral density, dichroic, etc.) decoded with beam search (width >= 5) and TMM-resimulated.

### Sample Size

Evaluations use **n=10,000** validation samples (except 15 nm thick: n=7,221).

---

## 3. In-Distribution Results

The model is evaluated on data drawn from the same distribution as training (10 nm thickness steps, 1--20 layers).

### 3.1 In-Distribution: 10 nm Steps

**Data**: `eval_data/len_20_10nm/` | **n=10,000** | GT mean length: 13.7 layers

| Decoding | MSE | MAE | R^2 |
|---|---|---|---|
| Greedy | 0.00419 | 0.0272 | 0.945 |
| Beam Top-1 | 0.02258 | 0.0562 | 0.705 |
| **Oracle-best** | **0.00329** | **0.0238** | **0.957** |

The model achieves strong in-distribution performance with R^2 = 0.957 (oracle) and R^2 = 0.945 (greedy). Greedy decoding outperforms beam top-1, indicating that beam search score ranking does not correlate well with spectral fidelity; however, the oracle metric shows that beam search does surface better candidates -- the gap between greedy and oracle represents headroom available through improved re-ranking.

**Sequence length behaviour**: The model generates a mean of 10.1 layers (median 10) despite ground-truth mean of 13.7. The decoded length histogram shows a sharp spike at 9--10 layers (~61% of all samples), indicating the model has found a preferred operating length for spectral approximation.

---

## 4. Out-of-Distribution Results: Thickness Step

The model was tested on data with thickness step sizes it was **not trained on** (5 nm, 15 nm thick, 20 nm thick). All OOD datasets use fixed 20-layer designs.

### 4.1 OOD: 5 nm Thickness Steps

**Data**: `eval_data/len_20_5nm/` | **n=10,000** | GT mean length: 13.6 layers, GT mean cum. depth: 1,736 nm

| Decoding | MSE | MAE | R^2 |
|---|---|---|---|
| Greedy | 0.00473 | 0.0308 | 0.942 |
| Beam Top-1 | 0.01665 | 0.0490 | 0.797 |
| **Oracle-best** | **0.00373** | **0.0270** | **0.955** |

The model performs comparably to in-distribution data on 5 nm steps. The finer granularity produces shallower structures (mean cum. depth 1,736 nm vs. 3,486 nm for 10 nm) well within the training range. The model generates even shorter sequences here (mean 8.8 layers) and achieves its highest compression ratio.

### 4.2 OOD: 15 nm Steps -- Thick Designs Only (cum. depth >= 11,000 nm)

**Data**: `eval_data/len_20_15nm_thick/` | **n=7,221** | All 20 layers, GT mean cum. depth: 11,676 nm (range 11,010--13,455 nm)

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.00466** | **0.931** | **0.00381** | **0.943** |
| **Dynamic NTK** | 1.5 | **0.00466** | **0.931** | **0.00381** | **0.943** |
| NTK-aware | 1.5 | 0.01095 | 0.836 | 0.01054 | 0.843 |
| YaRN | 1.5 | 0.01265 | 0.811 | 0.01179 | 0.824 |

Despite ground-truth cumulative depths exceeding the training maximum (11,000--13,455 nm vs. 10,000 nm), the model achieves oracle R^2 = 0.943. This is possible because the model compresses 20-layer designs into ~12 layers with cumulative depth ~3,140 nm -- well within the training range. No scaling and Dynamic NTK produce identical results. Static methods (NTK, YaRN) degrade performance, though they remain above R^2 = 0.8.

### 4.3 OOD: 20 nm Steps -- Thick Designs Only (cum. depth >= 11,000 nm)

**Data**: `eval_data/len_20_20nm_thick/` | **n=10,000** | All 20 layers, GT mean cum. depth: 15,134 nm (range 11,220--17,920 nm)

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.00484** | **0.927** | **0.00387** | **0.942** |
| **Dynamic NTK** | 2.0 | **0.00484** | **0.927** | **0.00387** | **0.942** |
| NTK-aware | 2.0 | 0.01121 | 0.832 | 0.01031 | 0.845 |
| YaRN | 2.0 | 0.01978 | 0.703 | 0.01750 | 0.737 |

Even with ground-truth cumulative depths up to ~18,000 nm (1.8x training max), the model maintains oracle R^2 = 0.942. The compression strategy is more aggressive here: the model maps 20-layer stacks (mean cum. depth 15,134 nm) to ~12-layer designs (mean cum. depth 3,278 nm) -- a 4.6x depth compression.

---

## 5. Out-of-Distribution Results: Sequence Length

The model was trained on sequences of 1--20 layers. To test generalization to longer designs, we evaluate on datasets with 10 nm thickness steps (matching training) but with sequence lengths of 20--30, 30--40, and 40--50 layers.

### 5.1 OOD: 20--30 Layers (10 nm Steps)

**Data**: `eval_data/len_30_10nm/` | **n=10,000** | GT mean length: 25.3 layers, GT mean cum. depth: 6,454 nm

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.00482** | **0.937** | **0.00393** | **0.948** |
| **Dynamic NTK** | 1.5 | **0.00482** | **0.937** | **0.00393** | **0.948** |
| PI | 1.5 | 0.01572 | 0.793 | 0.01533 | 0.798 |
| NTK-aware | 1.5 | 0.01592 | 0.791 | 0.01565 | 0.794 |
| YaRN | 1.5 | 0.01600 | 0.790 | 0.01554 | 0.796 |

### 5.2 OOD: 30--40 Layers (10 nm Steps)

**Data**: `eval_data/len_40_10nm/` | **n=10,000** | GT mean length: 35.4 layers, GT mean cum. depth: 9,031 nm

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.00487** | **0.935** | **0.00406** | **0.946** |
| **Dynamic NTK** | 2.0 | **0.00487** | **0.935** | **0.00406** | **0.946** |
| PI | 2.0 | 0.01620 | 0.783 | 0.01571 | 0.790 |
| NTK-aware | 2.0 | 0.01600 | 0.786 | 0.01562 | 0.791 |
| YaRN | 2.0 | 0.01789 | 0.760 | 0.01685 | 0.774 |

### 5.3 OOD: 40--50 Layers (10 nm Steps)

**Data**: `eval_data/len_50_10nm/` | **n=10,000** | GT mean length: 45.2 layers, GT mean cum. depth: 11,564 nm

| RoPE Method | Scale Factor | Greedy MSE | Greedy R^2 | Oracle MSE | Oracle R^2 |
|---|---|---|---|---|---|
| **None (no scaling)** | -- | **0.00484** | **0.935** | **0.00396** | **0.947** |
| **Dynamic NTK** | 2.5 | **0.00484** | **0.935** | **0.00396** | **0.947** |
| NTK-aware | 2.5 | 0.01616 | 0.784 | 0.01581 | 0.788 |
| PI | 2.5 | 0.01781 | 0.761 | 0.01702 | 0.772 |
| YaRN | 2.5 | 0.02368 | 0.683 | 0.02040 | 0.727 |

### 5.4 Sequence-Length OOD Analysis

**Performance is remarkably stable across all three length ranges.** Oracle R^2 is 0.948, 0.946, and 0.947 for len 30, 40, and 50 respectively -- essentially flat. This is only a modest drop from the in-distribution result (0.957).

The model's autoregressive decoder caps out at ~11 layers regardless of ground-truth length, effectively learning a compressed spectral approximation using fewer layers than the ground truth. The compression ratio grows with input length:

| Ground-truth layers | Greedy decoded layers (mean) | Compression ratio | Greedy R^2 | Oracle R^2 |
|---|---|---|---|---|
| 1--20 (in-dist) | 10.1 | 1.35x | 0.945 | 0.957 |
| 20--30 | 11.0 | 2.30x | 0.937 | 0.948 |
| 30--40 | 11.0 | 3.22x | 0.935 | 0.946 |
| 40--50 | 11.0 | 4.12x | 0.935 | 0.947 |

**RoPE scaling results**: The same pattern holds as in the thickness-step OOD experiments. No scaling and Dynamic NTK are tied and best. Static methods (NTK, YaRN, PI) all degrade performance significantly. Notably, even though ground-truth cumulative depths reach 2.5x the training maximum, Dynamic NTK still matches the unscaled baseline -- this is because the model's decoded sequences stay within ~11 layers, so the actual RoPE positions used during inference remain within the training range.

**RoPE scaling inflates decoded sequence length**: An interesting side-effect of static RoPE scaling is that it causes the model to generate longer sequences. At len 50, PI produces mean decoded length of 17.5 (vs. 11.0 for no-scale), NTK produces 15.2, and YaRN produces 12.0. Longer decoded sequences do not improve quality -- they correlate with *worse* R^2, suggesting the scaling distorts the model's learned stopping criterion.

---

## 6. Thickness Distribution Analysis

Across all conditions, the model exhibits a strong thickness mode at **190--200 nm**, accounting for 20--39% of all predicted layer thicknesses regardless of ground-truth distribution. This contrasts with ground-truth data which has a nearly uniform thickness distribution across the 10--500 nm range.

### Thickness Prediction Patterns

| Condition | GT mean thk (nm) | Greedy mean thk (nm) | Peak bin | Peak fraction | Greedy cum. depth (nm) | GT cum. depth (nm) | Depth compression |
|---|---|---|---|---|---|---|---|
| 10 nm (in-dist) | 254.6 | 215.4 | 190--200 | 32.4% | 2,185 | 3,486 | 1.6x |
| 5 nm (OOD) | 127.8 | 182.8 | 190--200 | 39.2% | 1,610 | 1,736 | 1.1x |
| 15 nm thick | 583.8 | 259.6 | 190--200 | 22.5% | 3,140 | 11,676 | 3.7x |
| 20 nm thick | 756.7 | 265.3 | 190--200 | 20.7% | 3,278 | 15,134 | 4.6x |
| 30 layers | 254.8 | 220.7 | 190--200 | 32.1% | 2,434 | 6,454 | 2.7x |
| 40 layers | 255.4 | 221.1 | 190--200 | 31.9% | 2,424 | 9,031 | 3.7x |
| 50 layers | 255.6 | 221.3 | 190--200 | 31.7% | 2,431 | 11,564 | 4.8x |

**Key observations**:

1. **Universal thickness mode**: The ~195 nm peak appears across all conditions, suggesting the model has learned that quarter-wave optical thicknesses near 195 nm are an effective building block for spectral reconstruction in the 400--1100 nm range. This is physically sensible: a 195 nm layer of a material with n ~ 2 (e.g., TiO2, Ta2O5, Si3N4) produces quarter-wave interference at ~1560 nm or half-wave near ~780 nm, and multilayer stacks built from these elements can produce a wide range of spectral features across the visible/NIR.

2. **Thickness upsampling for fine-step data**: For 5 nm inputs (GT mean 127.8 nm), the model *increases* predicted thicknesses to 182.8 nm. Combined with shorter sequences (8.8 vs 13.6 layers), the model remaps thin, many-layer designs into fewer, thicker layers.

3. **Stable output depth**: Greedy cumulative depth clusters around 2,000--3,300 nm regardless of input complexity, confirming the model operates in a fixed "output regime" for total optical depth.

---

## 7. Beam Search Re-ranking Analysis

Across all conditions, there is a consistent pattern in the relationship between greedy, top-1 beam, and oracle-best beam decoding:

| Condition | Greedy R^2 | Top-1 R^2 | Oracle R^2 | Greedy > Top-1 gap | Oracle headroom |
|---|---|---|---|---|---|
| 10 nm (in-dist) | 0.945 | 0.705 | 0.957 | +0.240 | +0.012 |
| 5 nm (OOD) | 0.942 | 0.797 | 0.955 | +0.145 | +0.012 |
| 15 nm thick | 0.931 | 0.610 | 0.943 | +0.321 | +0.013 |
| 20 nm thick | 0.927 | 0.621 | 0.942 | +0.307 | +0.015 |
| 30 layers | 0.937 | 0.661 | 0.948 | +0.276 | +0.012 |
| 40 layers | 0.935 | 0.676 | 0.946 | +0.259 | +0.011 |
| 50 layers | 0.935 | 0.677 | 0.947 | +0.259 | +0.012 |

**Key findings**:

1. **Beam scoring is poorly calibrated**: Greedy consistently beats beam top-1 by 0.15--0.32 R^2 points. The beam search log-probability score does not rank candidates by spectral fidelity.

2. **Oracle headroom is small but consistent**: The gap between greedy and oracle is +0.011 to +0.015 across all conditions. This means beam search generates candidates that are modestly better than greedy, but the improvement ceiling from re-ranking is limited (~1% R^2).

3. **5 nm data has the best-calibrated beam**: The greedy-to-top-1 gap is smallest (0.145) for 5 nm data, suggesting simpler/shallower structures produce beam candidates whose log-probabilities are more aligned with spectral quality.

4. **TMM re-ranking would close most of the gap**: Since oracle R^2 is only ~1% above greedy, the primary value of beam search would come from TMM-based re-scoring rather than generating fundamentally better designs. A practical system could use greedy decode for speed with minimal quality loss, or beam search + TMM re-ranking for the ~1% improvement.

---

## 8. Summary of Results

### Oracle R^2 Across All Conditions

| Eval Condition | Thickness Step | Layers | Cum. Depth Range | Oracle R^2 |
|---|---|---|---|---|
| 5 nm (OOD step) | 5 nm | 1--20 | ~250--2,500 nm | **0.955** |
| 10 nm (in-distribution) | 10 nm | 1--20 | ~100--7,500 nm | **0.957** |
| 15 nm, thick (>= 11,000 nm) | 15 nm | 20 | 11,010--13,455 nm | 0.943 |
| 20 nm, thick (>= 11,000 nm) | 20 nm | 20 | 11,220--17,920 nm | 0.942 |
| **20--30 layers (OOD length)** | 10 nm | 20--30 | ~2,000--15,000 nm | 0.948 |
| **30--40 layers (OOD length)** | 10 nm | 30--40 | ~3,000--20,000 nm | 0.946 |
| **40--50 layers (OOD length)** | 10 nm | 40--50 | ~4,000--25,000 nm | 0.947 |

### Key Takeaways

1. **Strong in-distribution performance**: The 13M model achieves oracle R^2 = 0.957 on 10 nm validation data, demonstrating effective inverse design of thin-film stacks up to 20 layers.

2. **Robust OOD generalisation**: Performance across all OOD conditions (different thickness steps, thick designs, longer sequences) stays within a narrow band of oracle R^2 = 0.942--0.957. This is a remarkably small degradation given that some conditions involve ground-truth structures 4.6x deeper or 2.5x longer than anything seen during training.

3. **Compression as a generalisation strategy**: The model's primary generalisation mechanism is design compression. Rather than reconstructing full-length ground-truth stacks, it generates shorter (~10--12 layer), shallower (~2,000--3,300 nm depth) designs that spectrally approximate the target. This compression ratio scales gracefully from 1.35x (in-distribution) to 4.8x (50-layer inputs) with minimal quality loss.

4. **Universal thickness mode at ~195 nm**: The model exhibits a strong preference for layer thicknesses around 190--200 nm regardless of the input distribution. This physically corresponds to quarter/half-wave optical thicknesses for common high-index materials in the visible/NIR range, suggesting the model has learned an efficient optical building-block strategy.

5. **RoPE scaling methods do not help**: Across all OOD conditions, no RoPE context extension method (PI, NTK, Dynamic NTK, YaRN) improved over the unscaled baseline. Dynamic NTK matched baseline (never activating since decoded positions stay in-range). Static methods actively harmed performance: NTK and YaRN reduced R^2 by 0.1--0.25; PI degraded more at higher scale factors. An unexpected side-effect is that static RoPE scaling inflates decoded sequence length (PI: up to 17.5 layers vs. 11.0 baseline at len 50), suggesting position distortion disrupts the model's stopping criterion.

6. **Beam search value is limited**: Greedy decoding outperforms beam top-1 by 0.15--0.32 R^2. Oracle-best beam candidates exceed greedy by only ~1%. The beam search scoring function is poorly calibrated for spectral fidelity. A practical deployment could rely on greedy decoding for speed, or use beam search with TMM re-ranking for a modest quality boost.

---

## 9. RoPE Scaling Methods Tested

For reference, the RoPE context extension methods evaluated:

| Method | Mechanism | Scale Factors Tested |
|---|---|---|
| **None** | Baseline -- no position scaling | -- |
| **PI** (Position Interpolation) | Divides all positions by scale factor | 1.5, 2.0, 2.5 |
| **NTK-aware** | Increases RoPE base frequency: `base * s^(d/(d-2))` | 1.5, 2.0, 2.5 |
| **Dynamic NTK** | Like NTK but only activates when max position exceeds training max (10,000 nm) | 1.5, 2.0, 2.5 |
| **YaRN** | NTK base scaling + attention temperature correction (`1/sqrt(s)`) | 1.5, 2.0, 2.5 |
