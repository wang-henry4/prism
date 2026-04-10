# Baselines

Comprehensive baseline evaluation for thin-film inverse design. All methods are evaluated on the same two benchmarks using identical metrics and TMM re-simulation.

## Evaluation Protocol

All methods follow the same evaluation pipeline:

1. **Input**: A target optical spectrum (142 floats: 71 reflectance + 71 transmittance values, sampled at 400–1100 nm in 10 nm steps).
2. **Output**: A predicted thin-film structure (list of materials and layer thicknesses in nm).
3. **Re-simulation**: The predicted structure is simulated using the Transfer Matrix Method (TMM) with incoherent substrate treatment (Glass substrate, 500 µm, s-polarization, normal incidence) to obtain a predicted spectrum.
4. **Metrics**: The re-simulated spectrum is compared against the target. We report MAE, MSE, and R² computed globally across all samples and wavelength points.

Importantly, predicted structures are **never** compared directly to ground-truth structures. Inverse design is a one-to-many problem — many different structures can produce the same optical response — so only spectral fidelity of the predicted design matters.

## Benchmarks

### Val Set (in-distribution)
10,000 randomly generated thin-film structures with 1–20 layers, 17 dielectric/metal/semiconductor materials, thicknesses 10–500 nm in 10 nm steps. Spectra computed via TMM. This tests generalization within the training distribution.

### Handcrafted Targets (out-of-distribution)
29 practical optical filter spectra representing real-world design tasks across 8 categories:
- **Narrowband filters** (3): laser line isolation at 532, 633, 850 nm
- **Broadband filters** (3): 100–150 nm FWHM bandpass, hard-edge bandpass
- **Edge filters** (6): steep longpass/shortpass with 10 nm transition widths
- **Notch filters** (4): laser line rejection at 532, 633, 785 nm; bandstop
- **Dichroic filters** (3): edge and bandpass dichroics with 5 nm transitions
- **Neutral density** (5): absorptive (OD 0.3–1.0) and reflective (OD 0.5–1.0)
- **Multi-bandpass** (2): RGB triple-band and dual fluorescence filters
- **Specialty** (3): hot mirror, cold mirror, linear variable filter

These targets are synthetically generated ideal spectra that do not appear in the training distribution, testing each method's ability to handle practical design specifications.

## Results

### Val Set (in-distribution)

| Method | Type | Samples | MAE ↓ | MSE ↓ | R² ↑ | Median MAE | P90 MAE |
|--------|------|---------|-------|-------|------|------------|---------|
| Simulated Annealing | Optimization | 1,000 | 0.0163 | 0.0017 | 0.978 | 0.0143 | 0.0365 |
| Differentiable TMM | Optimization | 1,000 | 0.0306 | 0.0037 | 0.954 | 0.0298 | 0.0566 |
| OptoGPT | Neural | 10,000 | 0.0585 | 0.0218 | 0.715 | 0.0311 | 0.1621 |
| Tandem Network | Neural | 10,000 | 0.0678 | 0.0175 | 0.771 | 0.0664 | 0.1224 |
| CVAE | Neural | 10,000 | 0.1612 | 0.0714 | 0.066 | 0.1302 | 0.3013 |

### Handcrafted Targets (out-of-distribution)

| Method | Type | Mean MAE ↓ | Mean MSE ↓ | Mean R² ↑ |
|--------|------|------------|------------|-----------|
| Simulated Annealing | Optimization | 0.1216 | 0.0450 | 0.788 |
| Differentiable TMM | Optimization | 0.1864 | 0.0863 | 0.573 |
| OptoGPT | Neural | 0.3499 | 0.2332 | -5.40 |
| CVAE | Neural | 0.3837 | 0.2697 | -1.24 |
| Tandem Network | Neural | 0.4297 | 0.2837 | -3.24 |

## Analysis

### Optimization vs. Neural Methods

Optimization baselines (SA, Diff TMM) substantially outperform all neural methods on both benchmarks. This is expected: they **optimize directly against each target spectrum at test time**, running hundreds to thousands of iterations with access to the exact physics simulator (TMM). Neural methods must generalize from training data and produce a design in a single forward pass.

Specific advantages of optimization methods:
- **Exact physics**: The TMM simulator serves as the optimization objective, introducing zero model approximation error.
- **Continuous search**: Thicknesses are optimized as continuous variables, avoiding the discretization artifacts present in token-based neural methods.
- **Flexible architecture**: SA can freely add, remove, and swap layers during search, exploring the full combinatorial design space for each target.

The tradeoff is inference speed: optimization methods are 3–4 orders of magnitude slower per sample.

### Neural Method Comparison

Among neural methods, OptoGPT achieves the best handcrafted target performance (MAE 0.350) despite being a pretrained model not specifically trained on our data distribution. Its autoregressive decoder can generate variable-length structures and benefits from the sequential generation process.

The Tandem Network achieves competitive val set performance (MAE 0.068, R² 0.771) but degrades severely on handcrafted targets (MAE 0.430), indicating overfitting to the training distribution. The forward consistency loss helps in-distribution but does not improve out-of-distribution generalization.

The CVAE performs worst overall (val MAE 0.161, R² 0.066). The single-sample decoding strategy (sampling one z from the prior) is suboptimal; a best-of-k strategy with TMM re-simulation would improve results but at the cost of inference speed.

### Key Observations

1. **Large OOD gap**: All neural methods show a 3–6× increase in MAE from val set to handcrafted targets, while optimization methods show only a 2–6× increase. Neural methods struggle particularly with narrowband filters (MAE ≈ 0.50), neutral density filters, and multi-bandpass designs.

2. **Negative R² on handcrafted targets**: All neural methods achieve negative mean R² on handcrafted targets, meaning their predictions are worse than predicting the global mean spectrum. This reflects complete failure on certain target categories.

3. **Speed-accuracy tradeoff**: There is a clear Pareto frontier — SA achieves the best accuracy at ~160s/sample, while neural methods achieve ~0.001–0.04s/sample at significantly lower accuracy. Closing this gap is the central challenge.

## Inference Speed

| Method | Time per sample | Speedup vs SA |
|--------|----------------|---------------|
| Tandem Network | ~0.001s | ~160,000× |
| CVAE | ~0.001s | ~160,000× |
| OptoGPT | ~0.04s | ~4,000× |
| Differentiable TMM | ~50s | ~3× |
| Simulated Annealing | ~160s | 1× |

## Method Details

### OptoGPT (Ma et al., 2024)

Pretrained decoder-only transformer for inverse design. Checkpoint from [huggingface.co/mataigao/optogpt](https://huggingface.co/mataigao/optogpt) (epoch 146).

- **Architecture**: 6-layer decoder with cross-attention to a spectrum memory token. d_model=1024, 8 attention heads, d_ff=512. 63.9M parameters.
- **Tokenization**: Joint material-thickness tokens (e.g., `TiO2_100`). Vocabulary of 904 tokens: 18 materials × 50 thickness buckets (10–500 nm, 10 nm steps) + BOS/EOS/UNK/PAD.
- **Decoding**: Greedy autoregressive decoding, max sequence length 22 (up to 20 layers).
- **Training**: Trained on the authors' original dataset with label-smoothed cross-entropy loss (smoothing=0.1) and Noam learning rate schedule.

### Differentiable TMM

PyTorch-differentiable implementation of the Transfer Matrix Method, enabling gradient-based inverse design via backpropagation through the physics simulator.

- **Optimization**: L-BFGS with strong Wolfe line search. Thicknesses parameterized in log-space to enforce positivity.
- **Multi-start**: 32 random restarts, each with a randomly sampled material sequence and layer count drawn from {3, 5, 7, 10, 14, 18}. Best result across all restarts is kept.
- **Iterations**: 300 L-BFGS iterations per restart (15 outer steps × 20 inner L-BFGS iterations).
- **No training required**: Optimizes directly against each target spectrum at test time.

### Simulated Annealing

Classical stochastic global optimization over the joint discrete-continuous design space.

- **State space**: Variable-length sequences of (material, thickness) pairs, 1–20 layers.
- **Move set**: Thickness perturbation with Gaussian noise σ=30 nm (50% probability), material swap (20%), layer insertion at random position (15%), layer removal (15%).
- **Schedule**: Exponential temperature decay from T=0.1 to T=1e-4 over 5,000 steps per restart.
- **Restarts**: 8 independent restarts with random initial designs. Best result across all restarts is kept.
- **No training required**: Optimizes directly against each target spectrum at test time.

### Tandem Network

Joint inverse-forward MLP architecture with spectrum consistency loss, following the tandem network paradigm for one-to-many inverse problems.

- **Inverse network** (793K params): 3-layer MLP (512 hidden units, ReLU). Input: 142-dim spectrum. Output: 20×17 material logits + 20 thickness predictions + 20-class layer count.
- **Forward network** (784K params): 3-layer MLP (512 hidden units, ReLU). Input: soft material one-hots + thicknesses + layer count (361-dim). Output: 142-dim spectrum with sigmoid activation.
- **Total**: 1.58M parameters.
- **Loss**: Material cross-entropy + thickness MSE + layer count cross-entropy + 10× spectrum reconstruction MSE (forward consistency).
- **Structure encoding**: Fixed-length representation — materials as one-hot vectors padded to 20 layers, thicknesses normalized by THK_MAX (500 nm), layer count as classification over 1–20.
- **Training**: Adam optimizer (lr=1e-3), cosine annealing schedule, batch size 4096, 30 epochs on 10M samples (~13 minutes on 1× NVIDIA L4). Early stopping on dev set spectrum reconstruction loss.

### CVAE (Conditional Variational Autoencoder)

Generative model that learns a distribution over designs conditioned on the target spectrum.

- **Encoder** (586K params): 2-layer MLP. Input: spectrum + structure (503-dim). Output: 64-dim mean and log-variance of the latent distribution.
- **Decoder** (826K params): 3-layer MLP (512 hidden units, ReLU). Input: spectrum + latent z (206-dim). Output: 20×17 material logits + 20 thickness predictions + 20-class layer count.
- **Total**: 1.41M parameters. Latent dimension: 64.
- **Loss**: Material cross-entropy + thickness MSE + layer count cross-entropy + β·KL divergence. KL weight β annealed linearly from 0 to 0.1 over the first 30% of training to prevent posterior collapse.
- **Training**: Adam optimizer (lr=1e-3), cosine annealing, batch size 4096, 30 epochs on 10M samples (~13 minutes on 1× NVIDIA L4).
- **Inference**: Sample z ~ N(0, I) and decode. Current results use a single sample; best-of-k with TMM re-simulation would improve accuracy at the cost of speed.

## Reproduce

```bash
# OptoGPT (requires pretrained checkpoint)
python baselines/eval_optogpt.py \
    --checkpoint /path/to/optogpt.pt \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow \
    --nk_dir ./nk --plot_dir ./plots/baselines/optogpt_10nm

# Differentiable TMM (no training, GPU recommended)
python baselines/eval_optim.py --method diff_tmm \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow \
    --nk_dir ./nk --n_val_samples 1000

# Simulated Annealing (no training, GPU recommended)
python baselines/eval_optim.py --method sa \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow \
    --nk_dir ./nk --n_val_samples 1000

# Tandem Network (trains from scratch, ~13 min on 1× L4)
python baselines/train_nn.py --model tandem \
    --train_path ./data/max_len_20_10nm/train \
    --dev_path ./data/max_len_20_10nm/dev/part_000.arrow \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow

# CVAE (trains from scratch, ~13 min on 1× L4)
python baselines/train_nn.py --model cvae \
    --train_path ./data/max_len_20_10nm/train \
    --dev_path ./data/max_len_20_10nm/dev/part_000.arrow \
    --val_path ./data/max_len_20_10nm/val/part_000.arrow
```

## References

- **OptoGPT**: Ma, T., Wang, H., & Guo, L. J. (2024). OptoGPT: A foundation model for inverse design in optical multilayer thin film structures. *Opto-Electronic Advances*, 7(7).
- **Tandem networks**: Liu, D., Tan, Y., Khoram, E., & Yu, Z. (2018). Training deep neural networks for the inverse design of nanophotonic structures. *ACS Photonics*, 5(4), 1365–1369.
- **CVAE for inverse design**: So, S., & Rho, J. (2019). Designing nanophotonic structures using conditional deep convolutional generative adversarial networks. *Nanophotonics*, 8(7), 1255–1261.
