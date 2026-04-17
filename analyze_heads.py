"""
Analyze material head confidence and thickness head agreement during decoding.

Performs step-by-step greedy decoding and records:
  1. Material head entropy, top-1 probability, and top1-top2 gap at each step
  2. Thickness head predictions for all materials vs. the chosen material

Usage:
    python analyze_heads.py \
        --checkpoint saved_models/inverse/inverse_v1/best.pt \
        --val_path ./data/val/part_000.arrow \
        --n_samples 1000 \
        --plot_dir ./plots/head_analysis
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from prism.constants import N_SPECTRUM
from prism.data.dataset import Vocab
from prism.model.prefix_material_thk_model import InverseModel


def _load_checkpoint(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def instrumented_greedy_decode(
    model: InverseModel,
    spectrum: torch.Tensor,   # [B, 142]
    vocab: Vocab,
    max_len: int = 101,
    device: torch.device | None = None,
) -> dict:
    """
    Greedy decode with per-step instrumentation of both heads.

    Returns dict with:
        mat_ids:      list[B] of list[int]
        thk_vals:     list[B] of list[float]
        step_entropy: list[B] of list[float]   — material head entropy per step
        step_top1_prob: list[B] of list[float] — P(argmax) per step
        step_top1_top2_gap: list[B] of list[float] — P(top1) - P(top2) per step
        step_top1_mat: list[B] of list[str]    — chosen material name per step
        step_top2_mat: list[B] of list[str]    — runner-up material name per step
        step_thk_chosen: list[B] of list[float] — thickness for chosen mat (nm)
        step_thk_all:    list[B] of list[dict]  — {mat_name: thk_nm} for real mats per step
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    B = spectrum.size(0)
    spectrum = spectrum.to(device)

    # Special token indices to exclude from "real material" analysis
    special_ids = {vocab.PAD, vocab.BOS, vocab.EOS}
    real_mat_ids = [i for i in range(len(vocab)) if i not in special_ids]
    real_mat_names = [vocab.decode(i) for i in real_mat_ids]

    mat_seqs = torch.full((B, 1), vocab.BOS, dtype=torch.long, device=device)
    thk_seqs = torch.zeros(B, 1, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    mat_results: list[list[int]]   = [[] for _ in range(B)]
    thk_results: list[list[float]] = [[] for _ in range(B)]

    step_entropy:        list[list[float]] = [[] for _ in range(B)]
    step_top1_prob:      list[list[float]] = [[] for _ in range(B)]
    step_top1_top2_gap:  list[list[float]] = [[] for _ in range(B)]
    step_top1_mat:       list[list[str]]   = [[] for _ in range(B)]
    step_top2_mat:       list[list[str]]   = [[] for _ in range(B)]
    step_thk_chosen:     list[list[float]] = [[] for _ in range(B)]
    step_thk_all:        list[list[dict]]  = [[] for _ in range(B)]

    with torch.no_grad():
        for _ in range(max_len - 1):
            T = mat_seqs.size(1)
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
            tgt_mask = causal.unsqueeze(0).expand(B, -1, -1)

            mat_logits, thk_pred = model(spectrum, mat_seqs, thk_seqs, tgt_mask)
            thk_nm = model.thk_to_nm(thk_pred)

            # Material head analysis
            logits_last = mat_logits[:, -1, :]          # [B, V]
            probs = F.softmax(logits_last, dim=-1)      # [B, V]
            log_probs = F.log_softmax(logits_last, dim=-1)

            entropy = -(probs * log_probs).sum(dim=-1)  # [B]
            top2_vals, top2_idx = probs.topk(2, dim=-1) # [B, 2]

            next_mat = logits_last.argmax(dim=-1)        # [B]

            # Thickness head analysis
            thk_last = thk_nm[:, -1, :]                  # [B, V]
            next_thk = thk_last.gather(-1, next_mat.unsqueeze(-1)).squeeze(-1)

            for b in range(B):
                if not finished[b]:
                    token = next_mat[b].item()
                    if token == vocab.EOS:
                        finished[b] = True
                    elif token != vocab.PAD:
                        mat_results[b].append(token)
                        thk_results[b].append(float(next_thk[b].item()))

                        step_entropy[b].append(float(entropy[b].item()))
                        step_top1_prob[b].append(float(top2_vals[b, 0].item()))
                        step_top1_top2_gap[b].append(
                            float((top2_vals[b, 0] - top2_vals[b, 1]).item())
                        )
                        step_top1_mat[b].append(vocab.decode(top2_idx[b, 0].item()))
                        step_top2_mat[b].append(vocab.decode(top2_idx[b, 1].item()))
                        step_thk_chosen[b].append(float(next_thk[b].item()))

                        # All real-material thicknesses at this step
                        thk_dict = {}
                        for mid, mname in zip(real_mat_ids, real_mat_names):
                            thk_dict[mname] = float(thk_last[b, mid].item())
                        step_thk_all[b].append(thk_dict)

            mat_seqs = torch.cat([mat_seqs, next_mat.unsqueeze(1)], dim=1)
            thk_seqs = torch.cat([thk_seqs, next_thk.unsqueeze(1)], dim=1)

            if finished.all():
                break

    return {
        "mat_ids": mat_results,
        "thk_vals": thk_results,
        "step_entropy": step_entropy,
        "step_top1_prob": step_top1_prob,
        "step_top1_top2_gap": step_top1_top2_gap,
        "step_top1_mat": step_top1_mat,
        "step_top2_mat": step_top2_mat,
        "step_thk_chosen": step_thk_chosen,
        "step_thk_all": step_thk_all,
    }


def plot_material_entropy(results: dict, save_path: str) -> None:
    """Plot material head entropy statistics by decoding step."""
    all_entropy = results["step_entropy"]
    max_steps = max(len(e) for e in all_entropy)

    # Gather per-step values
    per_step = [[] for _ in range(max_steps)]
    for seq in all_entropy:
        for t, v in enumerate(seq):
            per_step[t].append(v)

    steps = list(range(1, max_steps + 1))
    means = [np.mean(vs) for vs in per_step]
    medians = [np.median(vs) for vs in per_step]
    p25 = [np.percentile(vs, 25) for vs in per_step]
    p75 = [np.percentile(vs, 75) for vs in per_step]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, means, label="Mean", color="steelblue", linewidth=2)
    ax.plot(steps, medians, label="Median", color="tomato", linewidth=2)
    ax.fill_between(steps, p25, p75, alpha=0.2, color="steelblue", label="IQR")
    ax.set_xlabel("Decoding step")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Material Head Entropy per Step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_top1_prob(results: dict, save_path: str) -> None:
    """Plot top-1 probability and top1-top2 gap by decoding step."""
    all_top1 = results["step_top1_prob"]
    all_gap = results["step_top1_top2_gap"]
    max_steps = max(len(s) for s in all_top1)

    per_step_top1 = [[] for _ in range(max_steps)]
    per_step_gap = [[] for _ in range(max_steps)]
    for seq_t1, seq_gap in zip(all_top1, all_gap):
        for t, (v1, vg) in enumerate(zip(seq_t1, seq_gap)):
            per_step_top1[t].append(v1)
            per_step_gap[t].append(vg)

    steps = list(range(1, max_steps + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    means_t1 = [np.mean(vs) for vs in per_step_top1]
    medians_t1 = [np.median(vs) for vs in per_step_top1]
    p25_t1 = [np.percentile(vs, 25) for vs in per_step_top1]
    p75_t1 = [np.percentile(vs, 75) for vs in per_step_top1]

    ax1.plot(steps, means_t1, label="Mean", color="steelblue", linewidth=2)
    ax1.plot(steps, medians_t1, label="Median", color="tomato", linewidth=2)
    ax1.fill_between(steps, p25_t1, p75_t1, alpha=0.2, color="steelblue", label="IQR")
    ax1.set_xlabel("Decoding step")
    ax1.set_ylabel("P(top-1)")
    ax1.set_title("Top-1 Material Probability")
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    means_gap = [np.mean(vs) for vs in per_step_gap]
    medians_gap = [np.median(vs) for vs in per_step_gap]
    p25_gap = [np.percentile(vs, 25) for vs in per_step_gap]
    p75_gap = [np.percentile(vs, 75) for vs in per_step_gap]

    ax2.plot(steps, means_gap, label="Mean", color="steelblue", linewidth=2)
    ax2.plot(steps, medians_gap, label="Median", color="tomato", linewidth=2)
    ax2.fill_between(steps, p25_gap, p75_gap, alpha=0.2, color="steelblue", label="IQR")
    ax2.set_xlabel("Decoding step")
    ax2.set_ylabel("P(top-1) - P(top-2)")
    ax2.set_title("Top-1 vs Top-2 Probability Gap")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_entropy_histogram(results: dict, save_path: str) -> None:
    """Histogram of all per-step entropies across samples."""
    all_vals = [v for seq in results["step_entropy"] for v in seq]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_vals, bins=80, color="steelblue", edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Count")
    ax.set_title(f"Material Head Entropy Distribution (n={len(all_vals)} steps)")
    ax.axvline(np.median(all_vals), color="tomato", linestyle="--",
               label=f"Median={np.median(all_vals):.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_thickness_agreement(results: dict, save_path: str) -> None:
    """
    Analyze how much thickness predictions vary across materials.

    For each decoding step, compute the coefficient of variation (std/mean)
    of thickness predictions across all real materials.
    """
    all_thk_all = results["step_thk_all"]
    max_steps = max(len(s) for s in all_thk_all)

    per_step_cv = [[] for _ in range(max_steps)]
    per_step_std = [[] for _ in range(max_steps)]

    for seq in all_thk_all:
        for t, thk_dict in enumerate(seq):
            vals = np.array(list(thk_dict.values()))
            std = vals.std()
            mean = vals.mean()
            cv = std / mean if mean > 0 else 0.0
            per_step_cv[t].append(cv)
            per_step_std[t].append(std)

    steps = list(range(1, max_steps + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # CV plot
    means_cv = [np.mean(vs) for vs in per_step_cv]
    medians_cv = [np.median(vs) for vs in per_step_cv]
    p25_cv = [np.percentile(vs, 25) for vs in per_step_cv]
    p75_cv = [np.percentile(vs, 75) for vs in per_step_cv]

    ax1.plot(steps, means_cv, label="Mean", color="steelblue", linewidth=2)
    ax1.plot(steps, medians_cv, label="Median", color="tomato", linewidth=2)
    ax1.fill_between(steps, p25_cv, p75_cv, alpha=0.2, color="steelblue", label="IQR")
    ax1.set_xlabel("Decoding step")
    ax1.set_ylabel("CV (std / mean)")
    ax1.set_title("Thickness Prediction CV Across Materials")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Std plot (in nm)
    means_std = [np.mean(vs) for vs in per_step_std]
    medians_std = [np.median(vs) for vs in per_step_std]
    p25_std = [np.percentile(vs, 25) for vs in per_step_std]
    p75_std = [np.percentile(vs, 75) for vs in per_step_std]

    ax2.plot(steps, means_std, label="Mean", color="steelblue", linewidth=2)
    ax2.plot(steps, medians_std, label="Median", color="tomato", linewidth=2)
    ax2.fill_between(steps, p25_std, p75_std, alpha=0.2, color="steelblue", label="IQR")
    ax2.set_xlabel("Decoding step")
    ax2.set_ylabel("Std (nm)")
    ax2.set_title("Thickness Prediction Std Across Materials")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_thickness_heatmap(results: dict, save_path: str, n_examples: int = 5) -> None:
    """
    Heatmap of per-material thickness predictions for a few example samples.

    Each subplot shows one sample's decoding: rows = materials, cols = steps,
    cell color = predicted thickness. The chosen material at each step is marked.
    """
    all_thk_all = results["step_thk_all"]
    all_top1_mat = results["step_top1_mat"]

    # Pick samples with enough steps to be interesting
    candidates = [(i, len(seq)) for i, seq in enumerate(all_thk_all) if len(seq) >= 3]
    candidates.sort(key=lambda x: -x[1])
    selected = [c[0] for c in candidates[:n_examples]]

    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 1, figsize=(14, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, idx in zip(axes, selected):
        seq = all_thk_all[idx]
        chosen_mats = all_top1_mat[idx]
        n_steps = len(seq)
        mat_names = sorted(seq[0].keys())

        grid = np.zeros((len(mat_names), n_steps))
        for t, thk_dict in enumerate(seq):
            for m, mname in enumerate(mat_names):
                grid[m, t] = thk_dict[mname]

        im = ax.imshow(grid, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xticks(range(n_steps))
        ax.set_xticklabels([f"{t+1}\n({chosen_mats[t]})" for t in range(n_steps)],
                           fontsize=7, rotation=0)
        ax.set_yticks(range(len(mat_names)))
        ax.set_yticklabels(mat_names, fontsize=7)
        ax.set_xlabel("Step (chosen material)")
        ax.set_ylabel("Material")
        ax.set_title(f"Sample {idx} — Per-Material Thickness Predictions (nm)")

        # Mark chosen material
        for t in range(n_steps):
            chosen = chosen_mats[t]
            if chosen in mat_names:
                row = mat_names.index(chosen)
                ax.plot(t, row, "rx", markersize=10, markeredgewidth=2)

        plt.colorbar(im, ax=ax, label="Thickness (nm)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_chosen_vs_mean_thickness(results: dict, save_path: str) -> None:
    """
    Scatter: chosen material's thickness vs. mean thickness across all materials.

    If points cluster on y=x, the head predicts similar thickness regardless of
    material — i.e. it's mostly learning "how thick" not "how thick for this material".
    """
    chosen_thk = []
    mean_thk = []

    for seq_chosen, seq_all in zip(results["step_thk_chosen"], results["step_thk_all"]):
        for c, thk_dict in zip(seq_chosen, seq_all):
            chosen_thk.append(c)
            mean_thk.append(np.mean(list(thk_dict.values())))

    chosen_thk = np.array(chosen_thk)
    mean_thk = np.array(mean_thk)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(mean_thk, chosen_thk, s=2, alpha=0.3, color="steelblue")
    lim = max(chosen_thk.max(), mean_thk.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Mean thickness across all materials (nm)")
    ax.set_ylabel("Chosen material thickness (nm)")
    ax.set_title("Chosen vs. Mean Thickness Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    corr = np.corrcoef(mean_thk, chosen_thk)[0, 1]
    ax.text(0.05, 0.95, f"r = {corr:.4f}", transform=ax.transAxes,
            fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    import pyarrow.feather as feather

    parser = argparse.ArgumentParser(
        description="Analyze material and thickness head behavior during decoding"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_path", default="./data/val/part_000.arrow")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--plot_dir", default="./plots/head_analysis")
    parser.add_argument("--rope_scale_method", default="none",
                        choices=["none", "pi", "ntk", "dynamic_ntk", "yarn"])
    parser.add_argument("--rope_scale_factor", type=float, default=1.0)
    args = parser.parse_args()

    ckpt = _load_checkpoint(args.checkpoint)
    config = ckpt.get("config", {})
    vocab = Vocab()

    model = InverseModel(
        vocab_size=len(vocab),
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        dropout=config.get("dropout", 0.1),
        thk_head_hidden_layers=config.get("thk_head_hidden_layers", 2),
        rope_scale_method=args.rope_scale_method,
        rope_scale_factor=args.rope_scale_factor,
    )
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    table = feather.read_table(args.val_path, memory_map=True)
    all_spectra = table["spectra"].to_pylist()
    n_samples = min(args.n_samples, len(all_spectra))
    spectra = torch.tensor(all_spectra[:n_samples], dtype=torch.float32)

    # ── Instrumented greedy decode ────────────────────────────────────────
    decode_batch = 256
    combined: dict = {}

    print(f"Instrumented greedy decoding {n_samples} samples...")
    t0 = time.perf_counter()
    for start in range(0, n_samples, decode_batch):
        end = min(start + decode_batch, n_samples)
        batch_results = instrumented_greedy_decode(
            model, spectra[start:end], vocab, device=device,
        )
        if not combined:
            combined = {k: [] for k in batch_results}
        for k in batch_results:
            combined[k].extend(batch_results[k])

        elapsed = time.perf_counter() - t0
        sps = end / elapsed
        print(f"\r  {end}/{n_samples}  ({sps:.0f} s/s  ETA {(n_samples - end) / sps:.0f}s)",
              end="", flush=True)
    print(f"\r  {n_samples}/{n_samples} in {time.perf_counter() - t0:.1f}s")

    # ── Summary statistics ────────────────────────────────────────────────
    all_entropy = [v for seq in combined["step_entropy"] for v in seq]
    all_top1 = [v for seq in combined["step_top1_prob"] for v in seq]
    all_gap = [v for seq in combined["step_top1_top2_gap"] for v in seq]

    print(f"\n=== Material Head Confidence ===")
    print(f"  Entropy:       mean={np.mean(all_entropy):.4f}  "
          f"median={np.median(all_entropy):.4f}  "
          f"p90={np.percentile(all_entropy, 90):.4f}")
    print(f"  Top-1 prob:    mean={np.mean(all_top1):.4f}  "
          f"median={np.median(all_top1):.4f}  "
          f"p10={np.percentile(all_top1, 10):.4f}")
    print(f"  Top1-Top2 gap: mean={np.mean(all_gap):.4f}  "
          f"median={np.median(all_gap):.4f}  "
          f"p10={np.percentile(all_gap, 10):.4f}")

    # Thickness agreement
    all_cv = []
    all_std = []
    chosen_thk = []
    mean_thk = []
    for seq_chosen, seq_all in zip(combined["step_thk_chosen"], combined["step_thk_all"]):
        for c, thk_dict in zip(seq_chosen, seq_all):
            vals = np.array(list(thk_dict.values()))
            all_cv.append(vals.std() / vals.mean() if vals.mean() > 0 else 0.0)
            all_std.append(vals.std())
            chosen_thk.append(c)
            mean_thk.append(vals.mean())

    corr = np.corrcoef(mean_thk, chosen_thk)[0, 1]

    print(f"\n=== Thickness Head Agreement ===")
    print(f"  Cross-material CV:   mean={np.mean(all_cv):.4f}  "
          f"median={np.median(all_cv):.4f}")
    print(f"  Cross-material Std:  mean={np.mean(all_std):.1f} nm  "
          f"median={np.median(all_std):.1f} nm")
    print(f"  Chosen vs. mean thk: r={corr:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    os.makedirs(args.plot_dir, exist_ok=True)

    print(f"\nSaving plots to {args.plot_dir}/")
    plot_material_entropy(combined, os.path.join(args.plot_dir, "entropy_by_step.png"))
    plot_top1_prob(combined, os.path.join(args.plot_dir, "top1_prob_by_step.png"))
    plot_entropy_histogram(combined, os.path.join(args.plot_dir, "entropy_histogram.png"))
    plot_thickness_agreement(combined, os.path.join(args.plot_dir, "thickness_agreement.png"))
    plot_thickness_heatmap(combined, os.path.join(args.plot_dir, "thickness_heatmap.png"))
    plot_chosen_vs_mean_thickness(combined, os.path.join(args.plot_dir, "chosen_vs_mean_thk.png"))

    # ── Save JSON summary ─────────────────────────────────────────────────
    summary = {
        "n_samples": n_samples,
        "material_head": {
            "entropy": {
                "mean": float(np.mean(all_entropy)),
                "median": float(np.median(all_entropy)),
                "p10": float(np.percentile(all_entropy, 10)),
                "p90": float(np.percentile(all_entropy, 90)),
            },
            "top1_prob": {
                "mean": float(np.mean(all_top1)),
                "median": float(np.median(all_top1)),
                "p10": float(np.percentile(all_top1, 10)),
                "p90": float(np.percentile(all_top1, 90)),
            },
            "top1_top2_gap": {
                "mean": float(np.mean(all_gap)),
                "median": float(np.median(all_gap)),
                "p10": float(np.percentile(all_gap, 10)),
                "p90": float(np.percentile(all_gap, 90)),
            },
        },
        "thickness_head": {
            "cross_material_cv": {
                "mean": float(np.mean(all_cv)),
                "median": float(np.median(all_cv)),
            },
            "cross_material_std_nm": {
                "mean": float(np.mean(all_std)),
                "median": float(np.median(all_std)),
            },
            "chosen_vs_mean_correlation": float(corr),
        },
    }

    with open(os.path.join(args.plot_dir, "head_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.plot_dir}/head_analysis.json")


if __name__ == "__main__":
    main()
