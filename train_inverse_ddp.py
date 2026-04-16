"""
Train InverseModel with multi-GPU DDP support.

Usage:
    torchrun --nproc_per_node=4 train_inverse_ddp.py \
        --train_path ./data/max_len_20_10nm/train \
        --dev_path ./data/max_len_20_10nm/dev/part_000.arrow \
        --epochs 70 --batch_size 256 --run_name optoformer_v1
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from functools import partial
import glob

from prism.data.dataset import Vocab, ThinFilmDataset, _collate
from prism.model.prefix_material_thk_model import InverseModel
from prism.training.train import make_optimizer_and_scheduler, train_inverse


def make_ddp_dataloader(path, vocab, batch_size, num_workers, shuffle, rank, world_size):
    if os.path.isdir(path):
        parts = sorted(glob.glob(os.path.join(path, "part_*.arrow")))
        dataset = ConcatDataset([ThinFilmDataset(p, vocab) for p in parts])
    else:
        dataset = ThinFilmDataset(path, vocab)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(_collate, pad_id=vocab.PAD),
        pin_memory=True,
    ), sampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path",  default="./data/max_len_20_10nm/train")
    parser.add_argument("--dev_path",    default="./data/max_len_20_10nm/dev/part_000.arrow")
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
    parser.add_argument("--batch_size",  type=int,   default=256,
                        help="Per-GPU batch size. Effective batch = this × n_gpus")
    parser.add_argument("--run_name",    default="optoformer_v1")
    parser.add_argument("--save_dir",    default="./saved_models/inverse")
    parser.add_argument("--thk_head_hidden_layers", type=int, default=2)
    parser.add_argument("--thk_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset_schedule", action="store_true")
    args = parser.parse_args()

    # ── DDP init ──────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = (rank == 0)
    if is_main:
        print(f"DDP: {world_size} GPUs, per-GPU batch={args.batch_size}, "
              f"effective batch={args.batch_size * world_size}")

    vocab = Vocab()

    # ── Resume or fresh ───────────────────────────────────────────────────
    start_epoch = 1
    prior_loss_history = None

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        ckpt_config = ckpt.get("config", {})
        d_model = ckpt_config.get("d_model", args.d_model)
        n_layers = ckpt_config.get("n_layers", args.n_layers)
        n_heads = ckpt_config.get("n_heads", args.n_heads)
        d_ff = ckpt_config.get("d_ff", args.d_ff)
        dropout = ckpt_config.get("dropout", args.dropout)
        thk_head_hidden_layers = ckpt_config.get("thk_head_hidden_layers", args.thk_head_hidden_layers)
        start_epoch = ckpt["epoch"] + 1
        prior_loss_history = ckpt.get("loss_history", [])
        if is_main:
            print(f"Resuming from epoch {start_epoch}")
    else:
        d_model = args.d_model
        n_layers = args.n_layers
        n_heads = args.n_heads
        d_ff = args.d_ff
        dropout = args.dropout
        thk_head_hidden_layers = args.thk_head_hidden_layers

    config = dict(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, dropout=dropout,
        thk_head_hidden_layers=thk_head_hidden_layers,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    if is_main:
        print("Loading data...")
    train_loader, train_sampler = make_ddp_dataloader(
        args.train_path, vocab, args.batch_size, args.num_workers, True, rank, world_size
    )
    dev_loader, _ = make_ddp_dataloader(
        args.dev_path, vocab, args.batch_size, args.num_workers, False, rank, world_size
    )
    if is_main:
        print(f"Train: {len(train_loader.dataset):,} samples, "
              f"{len(train_loader)} batches/gpu/epoch")
        print(f"Dev: {len(dev_loader.dataset):,} samples")

    # ── Model ─────────────────────────────────────────────────────────────
    model = InverseModel(
        vocab_size=len(vocab),
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        d_ff=d_ff, dropout=dropout,
        thk_head_hidden_layers=thk_head_hidden_layers,
    ).to(device)

    if args.resume:
        model.load_state_dict(ckpt["model_state"])

    model = DDP(model, device_ids=[local_rank])

    if is_main:
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    total_steps = args.epochs * len(train_loader)
    if is_main:
        print(f"Total steps: {total_steps}  Warmup: {args.warmup_steps}")

    optimizer, scheduler = make_optimizer_and_scheduler(
        model, total_steps,
        peak_lr=args.peak_lr, warmup_steps=args.warmup_steps,
        min_lr=args.min_lr, weight_decay=args.weight_decay,
    )

    if args.resume and not args.reset_schedule:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    # ── Train ─────────────────────────────────────────────────────────────
    # Use the existing train_inverse but only save/log on rank 0.
    # We need to set the sampler epoch for proper shuffling.
    # Monkey-patch the train loop to set sampler epoch.

    _orig_train_loader_iter = train_loader.__iter__

    class EpochAwareLoader:
        """Wrapper that sets DistributedSampler epoch before each iteration."""
        def __init__(self, loader, sampler):
            self.loader = loader
            self.sampler = sampler
            self._epoch = 0

        def __iter__(self):
            self.sampler.set_epoch(self._epoch)
            self._epoch += 1
            return iter(self.loader)

        def __len__(self):
            return len(self.loader)

        @property
        def dataset(self):
            return self.loader.dataset

    epoch_train_loader = EpochAwareLoader(train_loader, train_sampler)

    # For DDP, pass the underlying module for saving, but train with DDP wrapper
    train_inverse(
        model, epoch_train_loader, dev_loader, optimizer, scheduler,
        args.epochs, device, args.save_dir, args.run_name,
        vocab_size=len(vocab), pad_id=vocab.PAD,
        config=config, vocab=vocab,
        thk_loss_weight=args.thk_loss_weight,
        start_epoch=start_epoch,
        prior_loss_history=prior_loss_history,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
