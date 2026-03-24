"""Matplotlib figure helpers for evaluation."""

import os

import matplotlib.pyplot as plt
import numpy as np

from optoformer.constants import WL_NM, N_WL


def plot_spectrum_comparison(
    pred: np.ndarray,
    target: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Side-by-side reflectance and transmittance comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(WL_NM, target[:N_WL], label="Target", color="steelblue")
    ax1.plot(WL_NM, pred[:N_WL],   label="Predicted", color="tomato", linestyle="--")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.set_title(f"{title} — Reflectance")

    ax2.plot(WL_NM, target[N_WL:], label="Target", color="steelblue")
    ax2.plot(WL_NM, pred[N_WL:],   label="Predicted", color="tomato", linestyle="--")
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Transmittance")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.set_title(f"{title} — Transmittance")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_loss_curve(loss_history: list[dict], save_path: str) -> None:
    """Train / dev loss curves."""
    epochs = [h["epoch"] for h in loss_history]
    train  = [h["train"] for h in loss_history]
    dev    = [h["dev"]   for h in loss_history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, label="Train")
    ax.plot(epochs, dev,   label="Dev")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Training Loss")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_loss_components(loss_history: list[dict], save_path: str) -> None:
    """Plot per-token mat vs thk loss to visualise scale difference."""
    if not loss_history or "dev_mat" not in loss_history[0]:
        return

    epochs = [h["epoch"] for h in loss_history]
    thk_w  = loss_history[0].get("thk_loss_weight", 1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw per-token component losses (dev)
    ax1.plot(epochs, [h["dev_mat"] for h in loss_history], label="Mat (KL)", color="steelblue")
    ax1.plot(epochs, [h["dev_thk"] for h in loss_history], label="Thk (MSE)", color="tomato")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Per-token loss")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title("Dev loss components (unweighted)")

    # Right: ratio thk/mat over time
    ratios = [h["dev_thk"] / (h["dev_mat"] + 1e-10) for h in loss_history]
    ax2.plot(epochs, ratios, color="mediumpurple")
    ax2.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Thk / Mat ratio")
    ax2.set_yscale("log")
    ax2.set_title(f"Loss ratio (thk_loss_weight={thk_w})")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_grad_stats(loss_history: list[dict], save_path: str) -> None:
    """Gradient norm and max plots over training epochs."""
    if not loss_history or "grad_norm_mean" not in loss_history[0]:
        return

    epochs = [h["epoch"] for h in loss_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Grad norm
    ax1.plot(epochs, [h["grad_norm_mean"] for h in loss_history],
             label="Mean", color="steelblue")
    ax1.fill_between(
        epochs,
        [h["grad_norm_mean"] for h in loss_history],
        [h["grad_norm_max"] for h in loss_history],
        alpha=0.2, color="steelblue", label="Max",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gradient L2 Norm")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title("Gradient Norm")

    # Grad max
    ax2.plot(epochs, [h["grad_max_mean"] for h in loss_history],
             label="Mean", color="tomato")
    ax2.fill_between(
        epochs,
        [h["grad_max_mean"] for h in loss_history],
        [h["grad_max_max"] for h in loss_history],
        alpha=0.2, color="tomato", label="Max",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Max |grad|")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.set_title("Max Absolute Gradient")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_design_comparison(
    pred_spectrum: np.ndarray,
    target_spectrum: np.ndarray,
    pred_materials: list[str],
    pred_thicknesses: list[float],
    target_materials: list[str],
    target_thicknesses: list[float],
    title: str,
    save_path: str,
) -> None:
    """
    Combined design + spectrum comparison for inverse model evaluation.

    Top row: thin-film stack visualisation (target vs predicted as stacked bars).
    Bottom row: reflectance and transmittance comparison.
    """
    import matplotlib.colors as mcolors

    all_mats = sorted(set(pred_materials + target_materials))
    cmap = plt.cm.tab20
    mat_colors = {m: mcolors.to_hex(cmap(i / max(len(all_mats), 1))) for i, m in enumerate(all_mats)}

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

    # ── Top: film stack comparison ────────────────────────────────────────────
    ax_stack = fig.add_subplot(gs[0, :])

    def _draw_stack(ax, materials, thicknesses, y_center, bar_height):
        x = 0.0
        for mat, thk in zip(materials, thicknesses):
            ax.barh(y_center, thk, left=x, height=bar_height,
                    color=mat_colors[mat], edgecolor="black", linewidth=0.5)
            if thk > 15:
                ax.text(x + thk / 2, y_center, f"{mat}\n{thk:.0f}",
                        ha="center", va="center", fontsize=7)
            x += thk

    _draw_stack(ax_stack, target_materials, target_thicknesses, 1.0, 0.35)
    _draw_stack(ax_stack, pred_materials, pred_thicknesses, 0.0, 0.35)

    ax_stack.set_xlabel("Cumulative thickness (nm)")
    ax_stack.set_yticks([0.0, 1.0])
    ax_stack.set_yticklabels(["Predicted", "Target"])
    ax_stack.set_ylim(-0.5, 1.5)
    ax_stack.set_title(f"{title} — Film Stack")

    # ── Bottom left: reflectance ──────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.plot(WL_NM, target_spectrum[:N_WL], label="Target", color="steelblue")
    ax_r.plot(WL_NM, pred_spectrum[:N_WL], label="Predicted", color="tomato", linestyle="--")
    ax_r.set_xlabel("Wavelength (nm)")
    ax_r.set_ylabel("Reflectance")
    ax_r.set_ylim(0, 1)
    ax_r.legend()
    ax_r.set_title("Reflectance")

    # ── Bottom right: transmittance ───────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, 1])
    ax_t.plot(WL_NM, target_spectrum[N_WL:], label="Target", color="steelblue")
    ax_t.plot(WL_NM, pred_spectrum[N_WL:], label="Predicted", color="tomato", linestyle="--")
    ax_t.set_xlabel("Wavelength (nm)")
    ax_t.set_ylabel("Transmittance")
    ax_t.set_ylim(0, 1)
    ax_t.legend()
    ax_t.set_title("Transmittance")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_beam_candidates(
    candidates: list[dict],
    target_spectrum: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """
    Plot top-K beam search candidates against a target spectrum.

    Each candidate dict has: materials, thicknesses, spectrum (TMM re-simulated), score.

    Layout:
      - Top: stacked film bars for each candidate
      - Bottom left: reflectance (target + all candidates)
      - Bottom right: transmittance (target + all candidates)
    """
    import matplotlib.colors as mcolors

    K = len(candidates)
    if K == 0:
        return

    # Collect all materials for consistent colouring
    all_mats = sorted({m for c in candidates for m in c["materials"]})
    cmap = plt.cm.tab20
    mat_colors = {m: mcolors.to_hex(cmap(i / max(len(all_mats), 1))) for i, m in enumerate(all_mats)}

    fig = plt.figure(figsize=(16, 4 + 2 * K))
    gs = fig.add_gridspec(2, 2, height_ratios=[K, 2], hspace=0.4, wspace=0.3)

    # ── Top: film stacks ──────────────────────────────────────────────────
    ax_stack = fig.add_subplot(gs[0, :])

    def _draw_stack(ax, materials, thicknesses, y_center, bar_height, label=None):
        x = 0.0
        for i, (mat, thk) in enumerate(zip(materials, thicknesses)):
            bar = ax.barh(y_center, thk, left=x, height=bar_height,
                          color=mat_colors.get(mat, "lightgray"),
                          edgecolor="black", linewidth=0.5)
            if thk > 15:
                ax.text(x + thk / 2, y_center, f"{mat}\n{thk:.0f}",
                        ha="center", va="center", fontsize=6)
            x += thk
        if label:
            ax.text(-5, y_center, label, ha="right", va="center", fontsize=8)

    for i, c in enumerate(candidates):
        y = K - 1 - i
        score_str = f"{c['score']:.2f}" if c.get("score") is not None else ""
        mse_str = f"MSE={c['mse']:.4f}" if "mse" in c else ""
        label = f"#{i+1} ({score_str} {mse_str})"
        _draw_stack(ax_stack, c["materials"], c["thicknesses"], y, 0.6, label)

    ax_stack.set_xlabel("Cumulative thickness (nm)")
    ax_stack.set_yticks([])
    ax_stack.set_ylim(-0.5, K - 0.5)
    ax_stack.set_title(f"{title} — Top {K} Candidates")

    # ── Bottom: spectra ───────────────────────────────────────────────────
    candidate_colors = plt.cm.Set1(np.linspace(0, 1, max(K, 3)))

    ax_r = fig.add_subplot(gs[1, 0])
    ax_r.plot(WL_NM, target_spectrum[:N_WL], label="Target", color="black", linewidth=2)
    for i, c in enumerate(candidates):
        spec = np.array(c["spectrum"])
        ax_r.plot(WL_NM, spec[:N_WL], label=f"#{i+1}", color=candidate_colors[i],
                  linestyle="--", alpha=0.8)
    ax_r.set_xlabel("Wavelength (nm)")
    ax_r.set_ylabel("Reflectance")
    ax_r.set_ylim(0, 1)
    ax_r.legend(fontsize=7, ncol=2)
    ax_r.set_title("Reflectance")

    ax_t = fig.add_subplot(gs[1, 1])
    ax_t.plot(WL_NM, target_spectrum[N_WL:], label="Target", color="black", linewidth=2)
    for i, c in enumerate(candidates):
        spec = np.array(c["spectrum"])
        ax_t.plot(WL_NM, spec[N_WL:], label=f"#{i+1}", color=candidate_colors[i],
                  linestyle="--", alpha=0.8)
    ax_t.set_xlabel("Wavelength (nm)")
    ax_t.set_ylabel("Transmittance")
    ax_t.set_ylim(0, 1)
    ax_t.legend(fontsize=7, ncol=2)
    ax_t.set_title("Transmittance")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_scatter(
    pred: np.ndarray,
    target: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """Predicted vs target scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target.ravel(), pred.ravel(), s=1, alpha=0.3)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close(fig)
