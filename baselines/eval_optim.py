"""
Evaluate optimization-based baselines on val set and handcrafted targets.

Usage:
    # Differentiable TMM (fast, GPU)
    python baselines/eval_optim.py --method diff_tmm \
        --val_path ./data/max_len_20_10nm/val/part_000.arrow --nk_dir ./nk

    # Simulated Annealing
    python baselines/eval_optim.py --method sa \
        --val_path ./data/max_len_20_10nm/val/part_000.arrow --nk_dir ./nk

    # Needle Optimization (slow — handcrafted only by default)
    python baselines/eval_optim.py --method needle \
        --val_path ./data/max_len_20_10nm/val/part_000.arrow --nk_dir ./nk

    # All methods, handcrafted targets only (quick comparison)
    python baselines/eval_optim.py --method all --n_val_samples 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.feather as feather
import torch
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prism.constants import N_SPECTRUM
from prism.data.sim import load_nk, simulate
from prism.eval.metrics import SpectrumMetrics
from prism.eval.targets import HANDCRAFTED_TARGETS
from baselines.diff_tmm import _build_nk_tensor
from baselines.optim_baselines import (
    diff_tmm_optimize,
    simulated_annealing,
    needle_optimization,
    DesignResult,
)

_nk_dict = None


def _worker_init(nk_dir: str):
    global _nk_dict
    _nk_dict = load_nk(nk_dir)


def _simulate_one(args):
    materials, thicknesses = args
    if not materials:
        return [0.0] * N_SPECTRUM
    try:
        return simulate(materials, thicknesses, _nk_dict)
    except Exception:
        return [0.0] * N_SPECTRUM


METHOD_FNS = {
    "diff_tmm": diff_tmm_optimize,
    "sa": simulated_annealing,
    "needle": needle_optimization,
}

METHOD_NAMES = {
    "diff_tmm": "Differentiable TMM",
    "sa": "Simulated Annealing",
    "needle": "Needle Optimization",
}


def run_method(method_name, spectra, mat_nk, sub_nk, device, label=""):
    """Run an optimization method on a list of target spectra."""
    fn = METHOD_FNS[method_name]
    results = []
    t0 = time.perf_counter()

    for i, spec in enumerate(spectra):
        result = fn(np.array(spec), mat_nk, sub_nk, device=device)
        results.append(result)
        if (i + 1) % 10 == 0 or i == len(spectra) - 1:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(spectra) - i - 1) / rate if rate > 0 else 0
            print(f"  {label} {i+1}/{len(spectra)}  "
                  f"({rate:.1f} samples/s, ETA {eta:.0f}s, last MAE={result.mae:.6f})")

    total_time = time.perf_counter() - t0
    return results, total_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="diff_tmm",
                        choices=["diff_tmm", "sa", "needle", "all"])
    parser.add_argument("--val_path", default="./data/max_len_20_10nm/val/part_000.arrow")
    parser.add_argument("--nk_dir", default="./nk")
    parser.add_argument("--n_val_samples", type=int, default=1000,
                        help="Val samples to evaluate (0 to skip)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plot_dir", default="./plots/baselines")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    mat_nk, sub_nk = _build_nk_tensor(args.nk_dir)

    methods = list(METHOD_FNS.keys()) if args.method == "all" else [args.method]

    # Load val data
    val_spectra = []
    if args.n_val_samples > 0:
        table = feather.read_table(args.val_path, memory_map=True)
        n = min(args.n_val_samples, len(table))
        val_spectra = table["spectra"].to_pylist()[:n]
        print(f"Val set: {n} samples")

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {METHOD_NAMES[method]}")
        print(f"{'='*60}")

        out_dir = os.path.join(args.plot_dir, method)
        os.makedirs(out_dir, exist_ok=True)
        all_metrics = {"model": method, "decoding": "optimization"}

        # ── Val set ───────────────────────────────────────────────────
        if val_spectra:
            print(f"\nVal set ({len(val_spectra)} samples)...")
            results, val_time = run_method(
                method, val_spectra, mat_nk, sub_nk, device, label="val"
            )

            # TMM re-simulation with numpy (ground truth comparison)
            all_mats = [r.materials for r in results]
            all_thks = [r.thicknesses for r in results]

            with Pool(args.workers, initializer=_worker_init, initargs=(args.nk_dir,)) as pool:
                pred_spectra = pool.map(_simulate_one, zip(all_mats, all_thks))

            pred_arr = np.array(pred_spectra)
            target_arr = np.array(val_spectra)
            metrics = SpectrumMetrics.compute(pred_arr, target_arr)
            per_sample_mae = np.mean(np.abs(pred_arr - target_arr), axis=1)
            metrics["median_mae"] = float(np.median(per_sample_mae))
            metrics["p90_mae"] = float(np.percentile(per_sample_mae, 90))
            metrics["time_s"] = val_time

            print(f"Val: MAE={metrics['mae']:.6f}  MSE={metrics['mse']:.6f}  "
                  f"R²={metrics['r2']:.4f}  ({val_time:.0f}s)")

            all_metrics.update(metrics)
            all_metrics["n_val_samples"] = len(val_spectra)

        # ── Handcrafted targets ───────────────────────────────────────
        if HANDCRAFTED_TARGETS:
            hc_spectra = [t["spectrum"] for t in HANDCRAFTED_TARGETS]
            print(f"\nHandcrafted targets ({len(hc_spectra)})...")
            results, hc_time = run_method(
                method, hc_spectra, mat_nk, sub_nk, device, label="hc"
            )

            all_mats = [r.materials for r in results]
            all_thks = [r.thicknesses for r in results]

            with Pool(args.workers, initializer=_worker_init, initargs=(args.nk_dir,)) as pool:
                pred_spectra = pool.map(_simulate_one, zip(all_mats, all_thks))

            hc_metrics_all = []
            for i, (r, sim_spec, target) in enumerate(
                zip(results, pred_spectra, HANDCRAFTED_TARGETS)
            ):
                target_spec = np.array(target["spectrum"])
                pred_spec = np.array(sim_spec)
                m = SpectrumMetrics.compute(
                    pred_spec.reshape(1, -1), target_spec.reshape(1, -1)
                )
                m["name"] = target["name"]
                m["label"] = target["label"]
                m["materials"] = r.materials
                m["thicknesses"] = r.thicknesses
                hc_metrics_all.append(m)

                print(f"  {target['label']:45s}  MAE={m['mae']:.6f}  R²={m['r2']:.4f}  "
                      f"layers={len(r.materials)}")

            hc_maes = [m["mae"] for m in hc_metrics_all]
            hc_mses = [m["mse"] for m in hc_metrics_all]
            hc_r2s = [m["r2"] for m in hc_metrics_all]
            print(f"\nHandcrafted summary: MAE={np.mean(hc_maes):.6f}  "
                  f"MSE={np.mean(hc_mses):.6f}  R²={np.mean(hc_r2s):.4f}")

            all_metrics["handcrafted_mean_mae"] = float(np.mean(hc_maes))
            all_metrics["handcrafted_mean_mse"] = float(np.mean(hc_mses))
            all_metrics["handcrafted_mean_r2"] = float(np.mean(hc_r2s))
            all_metrics["handcrafted"] = hc_metrics_all
            all_metrics["handcrafted_time_s"] = hc_time

        # ── Save ──────────────────────────────────────────────────────
        out_path = os.path.join(out_dir, "metrics.json")
        with open(out_path, "w") as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
