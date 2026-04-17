# PRISM 44M Head Analysis Report

## 1. Overview

This report analyzes the behavior of the material classification head and per-material thickness regression head during autoregressive greedy decoding on the 44M-parameter PRISM model. The goal is to understand why greedy decoding performs comparably to beam search oracle (TMM re-ranked) results, and why beam search top-1 (log-prob ranked) consistently underperforms both.

**Analysis script**: `analyze_heads.py`
**Results**: `eval_data/44m/head_analysis/`
**Samples**: 10,000 from the validation set

---

## 2. Material Head Confidence

### Summary Statistics

| Metric | Mean | Median | p10 | p90 |
|---|---|---|---|---|
| Entropy (nats) | 1.93 | 2.09 | 0.55 | 2.91 |
| Top-1 probability | 0.39 | 0.29 | 0.06 | 0.91 |
| Top1--Top2 gap | 0.28 | 0.06 | 0.0005 | 0.90 |

For reference, uniform entropy over 17 materials is ln(17) = 2.83 nats.

### Key Findings

**The material head is not confident.** Median top-1 probability is only 0.29 and median top1-top2 gap is 0.06. At most decoding steps the model barely distinguishes between candidate materials.

**Bimodal entropy distribution.** The entropy histogram shows two distinct modes:
- A sharp spike at ~0.4--0.5 nats: confident decisions, corresponding to the first 1--3 layers and EOS prediction.
- A large spike near ~2.85 nats (near-uniform): the model is effectively guessing among materials.

**Entropy increases with decoding depth.** Step 1 starts with entropy ~0.5 (the model knows how to begin a design). By step 5, entropy climbs to ~2.0--2.5 and stays elevated for all subsequent steps. This aligns with thin-film physics: early layers closest to the incident medium have the strongest spectral contribution and are most constrained, while deeper layers become increasingly interchangeable.

---

## 3. Thickness Head Agreement Across Materials

### Summary Statistics

| Metric | Mean | Median |
|---|---|---|
| Cross-material CV (std/mean) | 0.87 | 0.71 |
| Cross-material Std (nm) | 45.6 | 43.6 |
| Chosen vs. mean thickness correlation (r) | 0.12 | -- |

### Key Findings

**The thickness head makes material-specific predictions.** With a median CV of 0.71 (std is 71% of the mean thickness), the head predicts substantially different thicknesses for different materials at the same position. This is not a generic "how thick should this layer be" function.

**Chosen thickness is uncorrelated with the mean.** r = 0.12 between the chosen material's predicted thickness and the mean across all materials. The scatter plot shows a diffuse cloud rather than a tight diagonal. The head has learned that different materials require different thicknesses to achieve similar optical effects.

**Heatmaps confirm per-material structure.** Visual inspection of example samples shows clear material-dependent patterns: some materials consistently receive high thickness predictions while others receive low, and these patterns vary across decoding steps.

---

## 4. Implications for Greedy vs. Beam Search

### Observed performance ranking

```
Beam top-1 (log-prob ranked)  <  Greedy  ≈  Beam oracle (TMM re-ranked)
```

### Why beam search top-1 underperforms greedy

Beam search optimizes cumulative log-probability, which is a poor proxy for spectral quality when the material head is near-uniform. When entropy is ~2.85 nats, log-prob differences between materials are on the order of 0.01 nats. Beam search commits to sequences where no single step is the argmax but the cumulative score edges out alternatives. Over 10+ steps, these negligible differences compound into selecting a sequence the model doesn't actually prefer in any physically meaningful way.

Greedy always picks the step-wise argmax. Even when probabilities are near-uniform, the argmax is the model's best guess at each step. This myopic strategy is consistently taking the model's top choice rather than trading off one step's argmax for a negligibly better cumulative score.

### Why beam oracle recovers

Oracle re-ranking uses the actual physical objective (spectral MSE via TMM simulation), bypassing the broken log-prob ranking. The fact that oracle performance matches greedy confirms that beam search does explore useful candidates -- it just cannot identify the best one using its own scores.

### Why the problem is fundamentally degenerate

Many (material, thickness) combinations produce similar spectra. The near-uniform entropy at later steps is not a training failure -- it reflects genuine degeneracy in the inverse design problem. The thickness head compensates: whichever material greedy selects, the thickness head adapts accordingly, producing a physically reasonable design regardless of the material choice.

---

## 5. Architectural Takeaways

1. **The dual-head design works as intended.** The thickness head has internalized material-specific optical relationships (high CV, low chosen-vs-mean correlation). The shared backbone is learning the physics.

2. **The bottleneck is the material classifier, not the regressor.** The thickness head makes differentiated, material-conditioned predictions. The material head cannot discriminate beyond the first few layers.

3. **Log-prob is unreliable as a re-ranking signal.** For practical use, greedy decoding is preferred over beam search with log-prob ranking (faster, equal or better quality). If beam diversity is desired, external re-ranking (TMM simulation or a learned verifier) is required.

4. **Early layers dominate.** The model is confident only at steps 1--3. Improvements to later-layer material prediction would require either (a) breaking the degeneracy (e.g., imposing ordering constraints or material-pair preferences) or (b) accepting the degeneracy and focusing on thickness accuracy given any plausible material.

### Open question

Whether the material head's uncertainty at later steps is due to genuine physical degeneracy or exposure bias / error accumulation can be tested by comparing teacher-forced material accuracy vs. autoregressive accuracy at each position. If teacher-forced accuracy also drops at later positions, the degeneracy is real; if it stays high, the issue is error compounding during autoregressive decoding.

---

## 6. Plots

All plots saved to `eval_data/44m/head_analysis/`:

| File | Description |
|---|---|
| `entropy_by_step.png` | Material head entropy (mean, median, IQR) per decoding step |
| `top1_prob_by_step.png` | Top-1 probability and top1-top2 gap per step |
| `entropy_histogram.png` | Distribution of per-step entropy values across all samples |
| `thickness_agreement.png` | Cross-material thickness CV and std per step |
| `thickness_heatmap.png` | Per-material thickness predictions for 5 example samples |
| `chosen_vs_mean_thk.png` | Scatter of chosen vs. mean thickness (r = 0.12) |
| `head_analysis.json` | Machine-readable summary statistics |
