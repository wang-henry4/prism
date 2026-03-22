"""
Evaluate a saved ForwardModel or InverseModel checkpoint.

Usage — ForwardModel:
    python evaluate.py \\
        --checkpoint saved_models/forward/forward_best.pt \\
        --val_path ./data/val.arrow \\
        --plot_dir ./plots/forward_eval

Usage — InverseModel (with TMM re-simulation):
    python evaluate.py \\
        --checkpoint saved_models/inverse/inverse_best.pt \\
        --val_path ./data/val.arrow \\
        --nk_dir ./nk --n_samples 1000 \\
        --plot_dir ./plots/inverse_eval
"""

import argparse
import json
import os
import random

import numpy as np
import pyarrow.feather as feather
import torch

from optoformer.constants import N_SPECTRUM
from optoformer.data.dataset import Vocab, make_dataloader
from optoformer.eval.decode import greedy_decode
from optoformer.eval.metrics import SpectrumMetrics
from optoformer.eval.visualize import plot_design_comparison, plot_loss_curve, plot_scatter, plot_spectrum_comparison
from optoformer.model.transformer import make_forward_model, make_inverse_model


def _load_checkpoint(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_forward(args) -> None:
    ckpt   = _load_checkpoint(args.checkpoint)
    config = ckpt.get("config", {})
    vocab  = Vocab()

    model = make_forward_model(len(vocab), config)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dev_loader = make_dataloader(args.val_path, vocab, batch_size=256, shuffle=False, num_workers=0)

    all_pred, all_target = [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch.src_mat  = batch.src_mat.to(device)
            batch.src_thk  = batch.src_thk.to(device)
            batch.src_mask = batch.src_mask.to(device)
            pred = model(batch.src_mat, batch.src_thk, batch.src_mask)
            all_pred.append(pred.cpu().numpy())
            all_target.append(batch.spectrum.numpy())

    pred_arr   = np.concatenate(all_pred,   axis=0)
    target_arr = np.concatenate(all_target, axis=0)

    metrics = SpectrumMetrics.compute(pred_arr, target_arr)
    print(f"Forward eval  MSE={metrics['mse']:.6f}  MAE={metrics['mae']:.6f}  R²={metrics['r2']:.4f}")

    os.makedirs(args.plot_dir, exist_ok=True)

    n_plot = min(10, len(pred_arr))
    sample_indices = sorted(random.sample(range(len(pred_arr)), n_plot))
    for idx in sample_indices:
        plot_spectrum_comparison(
            pred_arr[idx], target_arr[idx],
            title=f"Sample {idx}",
            save_path=os.path.join(args.plot_dir, f"sample_{idx}.png"),
        )

    plot_scatter(
        pred_arr, target_arr,
        title=f"Forward  R²={metrics['r2']:.4f}",
        save_path=os.path.join(args.plot_dir, "scatter.png"),
    )

    if ckpt.get("loss_history"):
        plot_loss_curve(ckpt["loss_history"], save_path=os.path.join(args.plot_dir, "loss_curve.png"))

    with open(os.path.join(args.plot_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _tmm_worker_init(nk_dir: str) -> None:
    global _nk_dict
    from optoformer.data.sim import load_nk
    _nk_dict = load_nk(nk_dir)


def _tmm_simulate_one(args: tuple[list[str], list[float]]) -> list[float]:
    from optoformer.data.sim import simulate
    materials, thicknesses = args
    if not materials:
        return [0.0] * N_SPECTRUM
    try:
        return simulate(materials, thicknesses, _nk_dict)  # type: ignore[arg-type]
    except Exception:
        return [0.0] * N_SPECTRUM


def evaluate_inverse(args) -> None:
    from multiprocessing import Pool

    ckpt   = _load_checkpoint(args.checkpoint)
    config = ckpt.get("config", {})
    vocab  = Vocab()

    model = make_inverse_model(len(vocab), config)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    table      = feather.read_table(args.val_path, memory_map=True)
    n_samples  = min(args.n_samples, len(table))
    spectra_gt = torch.tensor(table["spectra"].to_pylist()[:n_samples], dtype=torch.float32)
    mats_gt    = table["materials"].to_pylist()[:n_samples]
    thks_gt    = table["thicknesses"].to_pylist()[:n_samples]

    # ── Phase 1: GPU decode (batched) ─────────────────────────────────────────
    decode_batch_size = 512
    all_pred_mats, all_pred_thks = [], []

    print(f"Decoding {n_samples} samples...")
    for start in range(0, n_samples, decode_batch_size):
        end        = min(start + decode_batch_size, n_samples)
        spec_batch = spectra_gt[start:end]
        mat_ids_list, thk_list = greedy_decode(model, spec_batch, vocab, device=device)

        for mat_ids, thk_vals in zip(mat_ids_list, thk_list):
            mat_names = [vocab.decode(i) for i in mat_ids if i not in (vocab.PAD, vocab.BOS, vocab.EOS)]
            thk_nm    = [max(5.0, t) for t in thk_vals]
            all_pred_mats.append(mat_names)
            all_pred_thks.append(thk_nm)

    # ── Phase 2: TMM re-simulation (parallel CPU) ────────────────────────────
    print(f"Re-simulating {n_samples} structures with {args.workers} workers...")
    with Pool(
        processes=args.workers,
        initializer=_tmm_worker_init,
        initargs=(args.nk_dir,),
    ) as pool:
        all_pred_spectra = pool.map(
            _tmm_simulate_one, zip(all_pred_mats, all_pred_thks)
        )

    all_gt_spectra = spectra_gt[:n_samples].tolist()
    pred_arr   = np.array(all_pred_spectra)
    target_arr = np.array(all_gt_spectra)

    metrics = SpectrumMetrics.compute(pred_arr, target_arr)
    print(
        f"Inverse eval (TMM re-sim)  "
        f"MSE={metrics['mse']:.6f}  MAE={metrics['mae']:.6f}  R²={metrics['r2']:.4f}"
    )

    os.makedirs(args.plot_dir, exist_ok=True)

    n_plot = min(10, len(pred_arr))
    sample_indices = sorted(random.sample(range(len(pred_arr)), n_plot))
    for idx in sample_indices:
        plot_spectrum_comparison(
            pred_arr[idx], target_arr[idx],
            title=f"Sample {idx}",
            save_path=os.path.join(args.plot_dir, f"sample_{idx}.png"),
        )
        plot_design_comparison(
            pred_arr[idx], target_arr[idx],
            all_pred_mats[idx], all_pred_thks[idx],
            mats_gt[idx], [float(t) for t in thks_gt[idx]],
            title=f"Sample {idx}",
            save_path=os.path.join(args.plot_dir, f"design_{idx}.png"),
        )

    plot_scatter(
        pred_arr, target_arr,
        title=f"Inverse (TMM)  R²={metrics['r2']:.4f}",
        save_path=os.path.join(args.plot_dir, "scatter.png"),
    )

    if ckpt.get("loss_history"):
        plot_loss_curve(ckpt["loss_history"], save_path=os.path.join(args.plot_dir, "loss_curve.png"))

    with open(os.path.join(args.plot_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_path",   default="./data/val.arrow")
    parser.add_argument("--nk_dir",     default="./nk")
    parser.add_argument("--n_samples",  type=int, default=1000)
    parser.add_argument("--workers",   type=int, default=8)
    parser.add_argument("--plot_dir",   default="./plots/eval")
    args = parser.parse_args()

    ckpt       = _load_checkpoint(args.checkpoint)
    is_inverse = any("spectrum_proj" in k for k in ckpt["model_state"])

    if is_inverse:
        evaluate_inverse(args)
    else:
        evaluate_forward(args)


if __name__ == "__main__":
    main()
