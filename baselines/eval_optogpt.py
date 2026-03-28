"""
Evaluate the pretrained OptoGPT checkpoint on optoformer val data.

Matches optoformer's evaluate.py pattern:
  1. Val set: greedy decode → TMM re-sim → MAE/MSE/R²
  2. Handcrafted targets: greedy decode → TMM re-sim → per-target metrics

Usage:
    python baselines/eval_optogpt.py \
        --checkpoint /path/to/optogpt.pt \
        --val_path ./data/max_len_20/val/part_000.arrow \
        --nk_dir ./nk
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pyarrow.feather as feather
import torch
from multiprocessing import Pool
from torch.autograd import Variable

# Add optogpt source to path so the checkpoint can unpickle its classes
OPTOGPT_SRC = os.path.join(os.path.dirname(__file__), "../../optogpt/optogpt")
sys.path.insert(0, OPTOGPT_SRC)

from core.models.transformer import make_model_I, subsequent_mask  # noqa: E402

# Reuse optoformer's sim and eval utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from optoformer.data.sim import load_nk, simulate  # noqa: E402
from optoformer.eval.metrics import SpectrumMetrics  # noqa: E402
from optoformer.eval.targets import HANDCRAFTED_TARGETS  # noqa: E402
from optoformer.eval.visualize import (  # noqa: E402
    plot_beam_candidates,
    plot_design_comparison,
    plot_scatter,
)

_nk_dict = None


def _worker_init(nk_dir: str):
    global _nk_dict
    _nk_dict = load_nk(nk_dir)


def _simulate_one(args):
    materials, thicknesses = args
    if not materials:
        return [0.0] * 142
    try:
        return simulate(materials, thicknesses, _nk_dict)
    except Exception:
        return [0.0] * 142


# ── Greedy decode ─────────────────────────────────────────────────────────────
def greedy_decode(model, spec_target, index_dict, word_dict, max_len, device):
    """Greedy autoregressive decode — returns list of token strings."""
    bos_id = word_dict["BOS"]
    ys = torch.ones(1, 1, dtype=torch.long, device=device).fill_(bos_id)
    src = torch.tensor(spec_target, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    tokens = []
    for _ in range(max_len - 1):
        trg_mask = subsequent_mask(ys.size(1)).to(device)
        out = model(src, ys, None, trg_mask)
        prob = model.generator(out[:, -1])
        next_id = prob.argmax(dim=-1).item()
        ys = torch.cat([ys, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        sym = index_dict[next_id]
        if sym == "EOS":
            break
        tokens.append(sym)
    return tokens


def parse_token(token):
    """'TiO2_100' -> ('TiO2', 100.0)"""
    parts = token.rsplit("_", 1)
    return parts[0], float(parts[1])


def decode_tokens(tokens):
    """Parse list of OptoGPT tokens into (materials, thicknesses)."""
    mats, thks = [], []
    for tok in tokens:
        try:
            m, t = parse_token(tok)
            mats.append(m)
            thks.append(t)
        except (ValueError, IndexError):
            pass
    return mats, thks


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_path", default="./data/max_len_20/val/part_000.arrow")
    parser.add_argument("--nk_dir", default="./nk")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plot_dir", default="./plots/baselines/optogpt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["configs"]

    model = make_model_I(
        cfg.spec_dim, cfg.struc_dim, cfg.layers,
        cfg.d_model, cfg.d_ff, cfg.head_num, cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    word_dict = cfg.struc_word_dict
    index_dict = cfg.struc_index_dict
    max_len = cfg.max_len
    print(f"Model: spec_dim={cfg.spec_dim}, vocab={cfg.struc_dim}, "
          f"layers={cfg.layers}, d_model={cfg.d_model}, max_len={max_len}")

    # ── Load val data ─────────────────────────────────────────────────────
    print("Loading val data...")
    table = feather.read_table(args.val_path, memory_map=True)
    spectra_gt = table["spectra"].to_pylist()
    mats_gt = table["materials"].to_pylist()
    thks_gt = table["thicknesses"].to_pylist()
    n_samples = len(spectra_gt) if args.n_samples is None else min(args.n_samples, len(spectra_gt))
    spectra_gt = spectra_gt[:n_samples]
    mats_gt = mats_gt[:n_samples]
    thks_gt = thks_gt[:n_samples]
    print(f"Evaluating {n_samples} samples")

    # ── Phase 1: val set decode ───────────────────────────────────────────
    print("Decoding val set (greedy)...")
    all_pred_mats, all_pred_thks = [], []
    t0 = time.perf_counter()

    with torch.no_grad():
        for i, spec in enumerate(spectra_gt):
            tokens = greedy_decode(model, spec, index_dict, word_dict, max_len, device)
            mats, thks = decode_tokens(tokens)
            all_pred_mats.append(mats)
            all_pred_thks.append(thks)

            if (i + 1) % 500 == 0:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / rate
                print(f"  {i+1}/{n_samples}  ({rate:.0f} samples/s, ETA {eta:.0f}s)")

    decode_time = time.perf_counter() - t0
    print(f"Decoded {n_samples} in {decode_time:.1f}s ({n_samples/decode_time:.0f} samples/s)")

    # ── Phase 2: TMM re-simulation ────────────────────────────────────────
    print(f"Re-simulating with {args.workers} workers...")
    with Pool(args.workers, initializer=_worker_init, initargs=(args.nk_dir,)) as pool:
        all_pred_spectra = pool.map(_simulate_one, zip(all_pred_mats, all_pred_thks))

    pred_arr = np.array(all_pred_spectra)
    target_arr = np.array(spectra_gt)

    metrics = SpectrumMetrics.compute(pred_arr, target_arr)
    per_sample_mae = np.mean(np.abs(pred_arr - target_arr), axis=1)
    metrics["median_mae"] = float(np.median(per_sample_mae))
    metrics["p90_mae"] = float(np.percentile(per_sample_mae, 90))

    print(f"Inverse eval (TMM re-sim)  "
          f"MAE={metrics['mae']:.6f}  MSE={metrics['mse']:.6f}  R²={metrics['r2']:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────
    os.makedirs(args.plot_dir, exist_ok=True)

    n_plot = min(10, len(pred_arr))
    sample_indices = sorted(random.sample(range(len(pred_arr)), n_plot))
    for idx in sample_indices:
        plot_design_comparison(
            pred_arr[idx], target_arr[idx],
            all_pred_mats[idx], all_pred_thks[idx],
            mats_gt[idx], [float(t) for t in thks_gt[idx]],
            title=f"Sample {idx}",
            save_path=os.path.join(args.plot_dir, f"design_{idx}.png"),
        )

    plot_scatter(
        pred_arr, target_arr,
        title=f"OptoGPT Inverse (TMM)  R²={metrics['r2']:.4f}",
        save_path=os.path.join(args.plot_dir, "scatter.png"),
    )

    # ── Phase 3: handcrafted targets ──────────────────────────────────────
    if HANDCRAFTED_TARGETS:
        print(f"\nEvaluating {len(HANDCRAFTED_TARGETS)} handcrafted targets (greedy)...")
        hc_results = []

        with torch.no_grad():
            for target in HANDCRAFTED_TARGETS:
                tokens = greedy_decode(
                    model, target["spectrum"], index_dict, word_dict, max_len, device,
                )
                mats, thks = decode_tokens(tokens)
                hc_results.append({"target": target, "materials": mats, "thicknesses": thks})

        # TMM re-simulate all handcrafted predictions
        hc_sim_jobs = [(r["materials"], r["thicknesses"]) for r in hc_results]
        with Pool(args.workers, initializer=_worker_init, initargs=(args.nk_dir,)) as pool:
            hc_sim_spectra = pool.map(_simulate_one, hc_sim_jobs)

        hc_dir = os.path.join(args.plot_dir, "handcrafted")
        os.makedirs(hc_dir, exist_ok=True)

        hc_metrics_all = []
        for r, sim_spec in zip(hc_results, hc_sim_spectra):
            target_spec = np.array(r["target"]["spectrum"])
            pred_spec = np.array(sim_spec)
            m = SpectrumMetrics.compute(pred_spec.reshape(1, -1), target_spec.reshape(1, -1))
            m["name"] = r["target"]["name"]
            m["label"] = r["target"]["label"]
            m["materials"] = r["materials"]
            m["thicknesses"] = r["thicknesses"]
            hc_metrics_all.append(m)

            print(f"  {r['target']['label']:40s}  MAE={m['mae']:.6f}  R²={m['r2']:.4f}  "
                  f"design={r['materials']}  thk={[f'{t:.0f}' for t in r['thicknesses']]}")

            # Wrap as a single candidate for plot_beam_candidates
            candidate = {
                "spectrum": sim_spec,
                "materials": r["materials"],
                "thicknesses": r["thicknesses"],
                "score": 0.0,
                "mse": m["mse"],
                "mae": m["mae"],
                "r2": m["r2"],
            }
            plot_beam_candidates(
                [candidate], target_spec,
                title=f"OptoGPT: {r['target']['label']}",
                save_path=os.path.join(hc_dir, f"{r['target']['name']}.png"),
            )

        # Summary
        hc_maes = [m["mae"] for m in hc_metrics_all]
        print(f"\nHandcrafted summary: mean_MAE={np.mean(hc_maes):.6f}  "
              f"median_MAE={np.median(hc_maes):.6f}")
        metrics["handcrafted_mean_mae"] = float(np.mean(hc_maes))
        metrics["handcrafted_median_mae"] = float(np.median(hc_maes))
        metrics["handcrafted"] = hc_metrics_all

    # ── Save ──────────────────────────────────────────────────────────────
    metrics["model"] = "optogpt"
    metrics["n_samples"] = n_samples
    metrics["decode_time_s"] = decode_time
    metrics["decoding"] = "greedy"

    output_path = os.path.join(args.plot_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
