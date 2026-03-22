"""
Compare two ForwardModel checkpoints on the dev set.

Usage:
    python compare.py \\
        --ckpt_v1 ../optogpt/saved_models/ol_transformer.pt \\
        --ckpt_v2 ./saved_models/forward/forward_best.pt \\
        --plot_dir ./plots/comparison
"""

import argparse
import json
import os

import numpy as np
import torch

from optoformer.data.dataset import Vocab, make_dataloader
from optoformer.eval.metrics import SpectrumMetrics
from optoformer.eval.visualize import plot_scatter, plot_spectrum_comparison
from optoformer.model.transformer import make_forward_model


def _load_model(ckpt_path: str, vocab: Vocab, device: torch.device):
    ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model  = make_forward_model(len(vocab), config)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def _collect_predictions(model, dev_loader, device):
    preds, targets = [], []
    with torch.no_grad():
        for batch in dev_loader:
            batch.src_mat  = batch.src_mat.to(device)
            batch.src_thk  = batch.src_thk.to(device)
            batch.src_mask = batch.src_mask.to(device)
            pred = model(batch.src_mat, batch.src_thk, batch.src_mask)
            preds.append(pred.cpu().numpy())
            targets.append(batch.spectrum.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two ForwardModel checkpoints")
    parser.add_argument("--ckpt_v1",    required=True)
    parser.add_argument("--ckpt_v2",    required=True)
    parser.add_argument("--val_path",   default="./data/val.arrow")
    parser.add_argument("--plot_dir",   default="./plots/comparison")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab  = Vocab()

    dev_loader = make_dataloader(
        args.val_path, vocab, args.batch_size, shuffle=False, num_workers=0
    )

    model_v1, ckpt_v1 = _load_model(args.ckpt_v1, vocab, device)
    model_v2, ckpt_v2 = _load_model(args.ckpt_v2, vocab, device)

    pred_v1, target = _collect_predictions(model_v1, dev_loader, device)
    pred_v2, _      = _collect_predictions(model_v2, dev_loader, device)

    m1 = SpectrumMetrics.compute(pred_v1, target)
    m2 = SpectrumMetrics.compute(pred_v2, target)

    print(f"v1  MSE={m1['mse']:.6f}  MAE={m1['mae']:.6f}  R²={m1['r2']:.4f}")
    print(f"v2  MSE={m2['mse']:.6f}  MAE={m2['mae']:.6f}  R²={m2['r2']:.4f}")

    os.makedirs(args.plot_dir, exist_ok=True)

    for i in range(min(4, len(pred_v1))):
        plot_spectrum_comparison(
            pred_v1[i], target[i],
            title=f"v1 sample {i}",
            save_path=os.path.join(args.plot_dir, f"v1_sample_{i}.png"),
        )
        plot_spectrum_comparison(
            pred_v2[i], target[i],
            title=f"v2 sample {i}",
            save_path=os.path.join(args.plot_dir, f"v2_sample_{i}.png"),
        )

    plot_scatter(pred_v1, target, f"v1  R²={m1['r2']:.4f}", os.path.join(args.plot_dir, "v1_scatter.png"))
    plot_scatter(pred_v2, target, f"v2  R²={m2['r2']:.4f}", os.path.join(args.plot_dir, "v2_scatter.png"))

    summary = {"v1": m1, "v2": m2}
    with open(os.path.join(args.plot_dir, "comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
