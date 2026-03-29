"""
Evaluate a saved InverseModel checkpoint with TMM re-simulation.

Usage:
    python evaluate.py \
        --checkpoint saved_models/inverse/inverse_best.pt \
        --val_path ./data/val.arrow \
        --nk_dir ./nk --n_samples 1000 \
        --plot_dir ./plots/inverse_eval
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pyarrow.feather as feather
import torch

from optoformer.constants import N_SPECTRUM
from optoformer.data.dataset import Vocab
from optoformer.eval.decode import beam_search_decode, beam_search_decode_topk, greedy_decode
from optoformer.eval.metrics import SpectrumMetrics
from optoformer.eval.targets import HANDCRAFTED_TARGETS
from optoformer.eval.visualize import plot_beam_candidates, plot_design_comparison, plot_grad_stats, plot_loss_components, plot_loss_curve, plot_scatter
from optoformer.model.prefix_material_thk_model import InverseModel


def _load_checkpoint(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


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


def main() -> None:
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(description="Evaluate a saved InverseModel checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_path",   default="./data/val.arrow")
    parser.add_argument("--nk_dir",     default="./nk")
    parser.add_argument("--n_samples",  type=int, default=1000)
    parser.add_argument("--beam_width", type=int, default=5,
                        help="Beam width for inverse decoding (1 = greedy)")
    parser.add_argument("--length_penalty", type=float, default=0.3,
                        help="Beam search length penalty (0=none, 1=full per-token)")
    parser.add_argument("--workers",   type=int, default=8)
    parser.add_argument("--plot_dir",   default="./plots/eval")
    args = parser.parse_args()

    ckpt   = _load_checkpoint(args.checkpoint)
    config = ckpt.get("config", {})
    vocab  = Vocab()

    model = InverseModel(
        vocab_size=len(vocab),
        d_model=config.get("d_model", 512),
        n_layers=config.get("n_layers", 6),
        n_heads=config.get("n_heads", 8),
        d_ff=config.get("d_ff", 2048),
        dropout=config.get("dropout", 0.1),
        thk_head_hidden_layers=config.get("thk_head_hidden_layers", 2),
        log_space_thk=config.get("log_space_thk", True),
    )
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    table      = feather.read_table(args.val_path, memory_map=True)
    n_samples  = min(args.n_samples, len(table))
    spectra_gt = torch.tensor(table["spectra"].to_pylist()[:n_samples], dtype=torch.float32)
    mats_gt    = table["materials"].to_pylist()[:n_samples]
    thks_gt    = table["thicknesses"].to_pylist()[:n_samples]

    # ── Phase 1: decode ──────────────────────────────────────────────────────
    decode_batch_size = 512
    all_pred_mats, all_pred_thks = [], []
    decode_fn = beam_search_decode if args.beam_width > 1 else greedy_decode

    strategy = f"beam search (width={args.beam_width})" if args.beam_width > 1 else "greedy"
    print(f"Decoding {n_samples} samples with {strategy}...")
    decode_start = time.perf_counter()

    for start in range(0, n_samples, decode_batch_size):
        end        = min(start + decode_batch_size, n_samples)
        spec_batch = spectra_gt[start:end]

        if args.beam_width > 1:
            mat_ids_list, thk_list = decode_fn(
                model, spec_batch, vocab, beam_width=args.beam_width,
                length_penalty=args.length_penalty, device=device,
            )
        else:
            mat_ids_list, thk_list = decode_fn(model, spec_batch, vocab, device=device)

        for mat_ids, thk_vals in zip(mat_ids_list, thk_list):
            mat_names = [vocab.decode(i) for i in mat_ids if i not in (vocab.PAD, vocab.BOS, vocab.EOS)]
            thk_nm    = [max(5.0, t) for t in thk_vals]
            all_pred_mats.append(mat_names)
            all_pred_thks.append(thk_nm)

        elapsed = time.perf_counter() - decode_start
        samples_per_sec = end / elapsed
        eta = (n_samples - end) / samples_per_sec if samples_per_sec > 0 else 0
        print(
            f"\r  decoded {end}/{n_samples}  "
            f"({samples_per_sec:.0f} samples/s  ETA {eta:.0f}s)",
            end="", flush=True,
        )

    decode_time = time.perf_counter() - decode_start
    print(f"\r  decoded {n_samples}/{n_samples} in {decode_time:.1f}s  "
          f"({n_samples / decode_time:.0f} samples/s)      ")

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

    # ── Phase 4: hand-crafted target spectra (top-K beam candidates) ─────
    if HANDCRAFTED_TARGETS:
        hc_beam_width = max(args.beam_width, 5)  # always show at least 5 candidates
        print(f"Evaluating {len(HANDCRAFTED_TARGETS)} hand-crafted targets "
              f"(top {hc_beam_width} candidates)...")

        hc_spectra = torch.tensor(
            [t["spectrum"] for t in HANDCRAFTED_TARGETS], dtype=torch.float32
        )

        # Get all top-K candidates per target
        all_topk = beam_search_decode_topk(
            model, hc_spectra, vocab, beam_width=hc_beam_width,
            length_penalty=args.length_penalty, device=device,
        )

        # Collect all candidates for batch TMM re-simulation
        sim_jobs: list[tuple[list[str], list[float]]] = []
        job_map: list[tuple[int, int]] = []  # (target_idx, candidate_idx)

        for i, candidates in enumerate(all_topk):
            for j, c in enumerate(candidates):
                mat_names = [vocab.decode(m) for m in c["mat_ids"]]
                thk_nm = [max(5.0, t) for t in c["thk_vals"]]
                c["materials"] = mat_names
                c["thicknesses"] = thk_nm
                sim_jobs.append((mat_names, thk_nm))
                job_map.append((i, j))

        with Pool(
            processes=args.workers,
            initializer=_tmm_worker_init,
            initargs=(args.nk_dir,),
        ) as pool:
            sim_results = pool.map(_tmm_simulate_one, sim_jobs)

        # Assign simulated spectra back to candidates
        for (i, j), sim_spec in zip(job_map, sim_results):
            all_topk[i][j]["spectrum"] = sim_spec

        hc_dir = os.path.join(args.plot_dir, "handcrafted")
        os.makedirs(hc_dir, exist_ok=True)

        for i, target in enumerate(HANDCRAFTED_TARGETS):
            target_spec = np.array(target["spectrum"])
            candidates = all_topk[i]

            # Compute per-candidate metrics
            for c in candidates:
                c_spec = np.array(c["spectrum"])
                c_metrics = SpectrumMetrics.compute(
                    c_spec.reshape(1, -1), target_spec.reshape(1, -1)
                )
                c["mse"] = c_metrics["mse"]
                c["mae"] = c_metrics["mae"]
                c["r2"] = c_metrics["r2"]

            print(f"\n  {target['label']}:")
            for j, c in enumerate(candidates):
                print(
                    f"    #{j+1}  score={c['score']:.2f}  "
                    f"MSE={c['mse']:.6f}  R²={c['r2']:.4f}  "
                    f"design={c['materials']}  "
                    f"thk={[f'{t:.0f}' for t in c['thicknesses']]}"
                )

            plot_beam_candidates(
                candidates, target_spec,
                title=target["label"],
                save_path=os.path.join(hc_dir, f"{target['name']}.png"),
            )

    if ckpt.get("loss_history"):
        plot_loss_curve(ckpt["loss_history"], save_path=os.path.join(args.plot_dir, "loss_curve.png"))
        plot_loss_components(ckpt["loss_history"], save_path=os.path.join(args.plot_dir, "loss_components.png"))
        plot_grad_stats(ckpt["loss_history"], save_path=os.path.join(args.plot_dir, "grad_stats.png"))

    with open(os.path.join(args.plot_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
