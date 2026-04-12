"""
RLVR post-training with GRPO (Group Relative Policy Optimization).

Fine-tunes a pretrained InverseModel using TMM re-simulation as a
verifiable reward signal. Directly optimizes for spectrum reconstruction
quality rather than proxy losses.

Usage:
    python train_rlvr.py \
        --checkpoint saved_models/inverse/inverse_v1/best.pt \
        --train_path ./data/train \
        --nk_dir ./nk \
        --n_samples 8 --temperature 1.0 --thk_noise_std 5.0 \
        --kl_coeff 0.1 --lr 1e-5 --batch_size 32 \
        --total_steps 5000 --eval_every 100 \
        --run_name rlvr_v1
"""

import argparse
import copy
import glob
import os

import numpy as np
import pyarrow.feather as feather
import torch

from prism.data.dataset import Vocab
from prism.eval.targets import HANDCRAFTED_TARGETS
from prism.model.prefix_material_thk_model import InverseModel
from prism.training.reward import TMMReward
from prism.training.rlvr import MixedSpectrumSource, train_rlvr


def _find_arrow_files(path: str) -> list[str]:
    """Resolve a path to a list of Arrow files."""
    if os.path.isdir(path):
        parts = sorted(glob.glob(os.path.join(path, "part_*.arrow")))
        if not parts:
            raise FileNotFoundError(f"No part_*.arrow files found in {path}")
        return parts
    return [path]


def main() -> None:
    parser = argparse.ArgumentParser(description="RLVR post-training with GRPO")

    # ── Checkpoint ──
    parser.add_argument("--checkpoint", required=True,
                        help="Path to pretrained InverseModel checkpoint")

    # ── Data ──
    parser.add_argument("--train_path", default="./data/train",
                        help="Path to training Arrow files (for spectrum sampling)")
    parser.add_argument("--dev_path", default="./data/dev",
                        help="Path to dev Arrow files (for periodic evaluation)")
    parser.add_argument("--nk_dir", default="./nk",
                        help="Directory with per-material nk CSV files")
    parser.add_argument("--handcrafted_ratio", type=float, default=0.5,
                        help="Fraction of each batch from handcrafted targets (0-1)")

    # ── GRPO hyperparams ──
    parser.add_argument("--n_samples", type=int, default=32,
                        help="Number of rollouts per spectrum (G)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Material sampling temperature")
    parser.add_argument("--thk_noise_std", type=float, default=5.0,
                        help="Thickness noise σ in nm")
    parser.add_argument("--kl_coeff", type=float, default=0.1,
                        help="KL penalty coefficient against reference policy")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping norm")

    # ── Optimizer ──
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Peak learning rate")
    parser.add_argument("--total_steps", type=int, default=5000,
                        help="Total GRPO training steps")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Linear warmup steps")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of spectra per batch (×G = total rollouts)")

    # ── Eval & output ──
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Steps between evaluations")
    parser.add_argument("--n_eval_samples", type=int, default=200,
                        help="Max dev samples for periodic evaluation")
    parser.add_argument("--reward_mode", default="normalized_mse",
                        choices=["normalized_mse", "r2", "neg_mse", "neg_mae"],
                        help="Reward function mode")
    parser.add_argument("--tmm_workers", type=int, default=8,
                        help="Number of TMM simulation workers")
    parser.add_argument("--save_dir", default="./saved_models/rlvr")
    parser.add_argument("--run_name", default="rlvr_v1")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = Vocab()

    # ── Load pretrained checkpoint ──
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    d_model = config.get("d_model", 512)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 8)
    d_ff = config.get("d_ff", 2048)
    dropout = config.get("dropout", 0.1)
    thk_head_hidden_layers = config.get("thk_head_hidden_layers", 2)

    model_kwargs = dict(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        thk_head_hidden_layers=thk_head_hidden_layers,
    )

    # Policy model (trainable)
    model = InverseModel(**model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Policy model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Reference model (frozen copy)
    ref_model = InverseModel(**model_kwargs).to(device)
    ref_model.load_state_dict(ckpt["model_state"])
    ref_model.eval()
    print("Reference model loaded (frozen)")

    # ── Data sources ──
    arrow_files = _find_arrow_files(args.train_path)
    print(f"Training spectra: {len(arrow_files)} Arrow files from {args.train_path}")

    data_source = MixedSpectrumSource(
        arrow_paths=arrow_files,
        handcrafted_targets=HANDCRAFTED_TARGETS,
        batch_size=args.batch_size,
        handcrafted_ratio=args.handcrafted_ratio,
    )
    print(f"Data: {data_source.n_arrow} Arrow spectra + {data_source.n_hc} handcrafted, "
          f"ratio={args.handcrafted_ratio}")

    # Dev spectra for evaluation
    eval_spectra = None
    if os.path.exists(args.dev_path):
        dev_files = _find_arrow_files(args.dev_path)
        dev_spectra_list = []
        for f in dev_files:
            table = feather.read_table(f, memory_map=True)
            dev_spectra_list.extend(table["spectra"].to_pylist())
        eval_spectra = torch.tensor(
            dev_spectra_list[:args.n_eval_samples], dtype=torch.float32
        )
        print(f"Eval spectra: {len(eval_spectra)} from {args.dev_path}")

    # ── Reward function ──
    reward_fn = TMMReward(
        nk_dir=args.nk_dir,
        n_workers=args.tmm_workers,
        reward_mode=args.reward_mode,
    )
    print(f"Reward: {args.reward_mode}, {args.tmm_workers} TMM workers")

    # ── Train ──
    full_config = {**config, "rlvr": vars(args)}

    step_history = train_rlvr(
        model=model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        data_source=data_source,
        vocab=vocab,
        n_samples=args.n_samples,
        temperature=args.temperature,
        thk_noise_std=args.thk_noise_std,
        kl_coeff=args.kl_coeff,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        eval_spectra=eval_spectra,
        save_dir=args.save_dir,
        run_name=args.run_name,
        device=device,
        config=full_config,
    )

    print(f"\nDone. Checkpoints saved to {os.path.join(args.save_dir, args.run_name)}/")


if __name__ == "__main__":
    main()
