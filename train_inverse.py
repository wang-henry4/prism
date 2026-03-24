"""
Train the InverseModel (target spectrum → thin-film structure).

Usage:
    python train_inverse.py \\
        --train_path ./data/train --dev_path ./data/dev \\
        --d_model 512 --n_layers 6 --n_heads 8 \\
        --epochs 200 --batch_size 256 --run_name inverse_v1
"""

import argparse

import torch

from optoformer.data.dataset import Vocab, make_dataloader
from optoformer.model.transformer import make_inverse_model
from optoformer.training.train import make_optimizer_and_scheduler, train_inverse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train InverseModel")
    parser.add_argument("--train_path",  default="./data/train")
    parser.add_argument("--dev_path",    default="./data/dev")
    parser.add_argument("--d_model",     type=int,   default=512)
    parser.add_argument("--n_layers",    type=int,   default=4)
    parser.add_argument("--n_heads",     type=int,   default=4) # d_model/n_heads should be at least 64
    parser.add_argument("--d_ff",        type=int,   default=2048) # d_ff should be at least 4*d_model for good performance
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--peak_lr",      type=float, default=3e-4)
    parser.add_argument("--min_lr",       type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int,   default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--run_name",    default="inverse_v1")
    parser.add_argument("--save_dir",    default="./saved_models/inverse")
    parser.add_argument("--arch",        default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--thk_head_hidden_layers", type=int, default=2,
                        help="Number of hidden layers in the per-material thickness MLP (arch D only)")
    parser.add_argument("--thk_loss_weight", type=float, default=0.001,
                        help="Weight applied to thickness MSE loss (use to balance vs material KL loss)")
    parser.add_argument("--num_workers", type=int,   default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = Vocab()
    config = dict(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        arch=args.arch,
    )
    if args.arch == "D":
        config["thk_head_hidden_layers"] = args.thk_head_hidden_layers
    print("initializing data loaders...")
    train_loader = make_dataloader(
        args.train_path, vocab, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    dev_loader = make_dataloader(
        args.dev_path, vocab, args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    print("data loaders ready")
    print("initializing model...")
    model = make_inverse_model(len(vocab), config).to(device)
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

    train_inverse(
        model, train_loader, dev_loader, optimizer, scheduler,
        args.epochs, device, args.save_dir, args.run_name,
        vocab_size=len(vocab), pad_id=vocab.PAD,
        config=config, vocab=vocab,
        thk_loss_weight=args.thk_loss_weight,
    )


if __name__ == "__main__":
    main()
