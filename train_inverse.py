"""
Train the InverseModel (target spectrum → thin-film structure).

Uses the Augmented Regression RoPE Thickness Encoding architecture:
spectrum prefix + causal self-attention + cumulative-depth RoPE +
per-material thickness MLP head.

Usage:
    # Fresh training
    python train_inverse.py \
        --train_path ./data/train --dev_path ./data/dev \
        --d_model 256 --n_layers 4 --n_heads 4 \
        --epochs 60 --batch_size 1024 --run_name inverse_v1

    # Resume from checkpoint
    python train_inverse.py \
        --resume saved_models/inverse/inverse_v1/latest.pt \
        --epochs 30 --run_name inverse_v1_continued
"""

import argparse

import torch

from optoformer.data.dataset import Vocab, make_dataloader
from optoformer.model.prefix_material_thk_model import InverseModel
from optoformer.training.train import make_optimizer_and_scheduler, train_inverse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InverseModel")
    parser.add_argument("--train_path",  default="./data/train")
    parser.add_argument("--dev_path",    default="./data/dev")
    parser.add_argument("--d_model",     type=int,   default=512)
    parser.add_argument("--n_layers",    type=int,   default=6)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--d_ff",        type=int,   default=2048)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--peak_lr",      type=float, default=3e-4)
    parser.add_argument("--min_lr",       type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int,   default=5500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",      type=int,   default=70)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--run_name",    default="inverse_v1")
    parser.add_argument("--save_dir",    default="./saved_models/inverse")
    parser.add_argument("--thk_head_hidden_layers", type=int, default=2,
                        help="Number of hidden layers in the per-material thickness MLP")
    parser.add_argument("--thk_loss_weight", type=float, default=1.0,
                        help="Weight applied to thickness MSE loss (use to balance vs material KL loss). "
                             "Default 1.0 assumes log-space thickness; use ~0.001 with --no_log_space_thk.")
    parser.add_argument("--no_log_space_thk", action="store_true",
                        help="Disable log-space thickness prediction (use raw nm instead of softplus → exp)")
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (e.g. saved_models/inverse/run/latest.pt)")
    parser.add_argument("--reset_schedule", action="store_true",
                        help="When resuming, reset epoch counter and LR schedule (fresh warmup + cosine from loaded weights)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = Vocab()

    # ── Resume or fresh start ─────────────────────────────────────────────
    start_epoch = 1
    prior_loss_history = None

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config", {})

        # Use checkpoint config for model architecture, CLI args for training
        d_model = ckpt_config.get("d_model", args.d_model)
        n_layers = ckpt_config.get("n_layers", args.n_layers)
        n_heads = ckpt_config.get("n_heads", args.n_heads)
        d_ff = ckpt_config.get("d_ff", args.d_ff)
        dropout = ckpt_config.get("dropout", args.dropout)
        thk_head_hidden_layers = ckpt_config.get("thk_head_hidden_layers", args.thk_head_hidden_layers)
        log_space_thk = ckpt_config.get("log_space_thk", True)

        start_epoch = ckpt["epoch"] + 1
        prior_loss_history = ckpt.get("loss_history", [])
        print(f"  Checkpoint epoch: {ckpt['epoch']}, resuming from epoch {start_epoch}")
    else:
        d_model = args.d_model
        n_layers = args.n_layers
        n_heads = args.n_heads
        d_ff = args.d_ff
        dropout = args.dropout
        thk_head_hidden_layers = args.thk_head_hidden_layers
        log_space_thk = not args.no_log_space_thk

    config = dict(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        thk_head_hidden_layers=thk_head_hidden_layers,
        log_space_thk=log_space_thk,
    )

    print("initializing data loaders...")
    train_loader = make_dataloader(
        args.train_path, vocab, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    dev_loader = make_dataloader(
        args.dev_path, vocab, args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print("data loaders ready")

    print("initializing model...")
    model = InverseModel(
        vocab_size=len(vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        thk_head_hidden_layers=thk_head_hidden_layers,
        log_space_thk=log_space_thk,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    total_steps = args.epochs * len(train_loader)
    print(f"Total steps: {total_steps}  Warmup steps: {args.warmup_steps}")
    if args.warmup_steps > total_steps * 0.5:
        print(
            f"WARNING: warmup_steps ({args.warmup_steps}) > 50% of total_steps ({total_steps}). "
            f"LR will barely change during training. Consider reducing --warmup_steps or increasing --epochs."
        )
    optimizer, scheduler = make_optimizer_and_scheduler(
        model, total_steps,
        peak_lr=args.peak_lr, warmup_steps=args.warmup_steps,
        min_lr=args.min_lr, weight_decay=args.weight_decay,
    )

    if args.resume:
        model.load_state_dict(ckpt["model_state"])
        if args.reset_schedule:
            start_epoch = 1
            prior_loss_history = None
            print(f"  Loaded model weights only (schedule and optimizer reset)")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            print(f"  Loaded model, optimizer, and scheduler state")

    train_inverse(
        model, train_loader, dev_loader, optimizer, scheduler,
        args.epochs, device, args.save_dir, args.run_name,
        vocab_size=len(vocab), pad_id=vocab.PAD,
        config=config, vocab=vocab,
        thk_loss_weight=args.thk_loss_weight,
        start_epoch=start_epoch,
        prior_loss_history=prior_loss_history,
    )


if __name__ == "__main__":
    main()
