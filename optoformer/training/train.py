"""
Training utilities: cosine-annealing LR schedule, LabelSmoothing loss, and
train_forward / train_inverse loops with checkpoint saving.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# ── Learning-rate schedule ─────────────────────────────────────────────────────

def make_optimizer_and_scheduler(
    model: nn.Module,
    total_steps: int,
    peak_lr: float = 3e-4,
    warmup_steps: int = 2000,
    min_lr: float = 1e-6,
    weight_decay: float = 0.01,
) -> tuple[AdamW, SequentialLR]:
    """
    Create AdamW optimizer with cosine annealing + linear warmup.

    Args:
        model:        Model whose parameters to optimise.
        total_steps:  Total number of training steps (epochs × batches_per_epoch).
        peak_lr:      Maximum learning rate after warmup.
        warmup_steps: Number of linear warmup steps.
        min_lr:       Minimum learning rate at end of cosine decay.
        weight_decay: AdamW weight decay.

    Returns:
        (optimizer, scheduler) — call scheduler.step() after each optimizer.step().
    """
    optimizer = AdamW(
        model.parameters(), lr=peak_lr,
        betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay,
    )
    warmup = LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
    )
    return optimizer, scheduler


# ── Loss functions ─────────────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Label-smoothed cross-entropy via KL divergence.
    smoothing=0.1 distributes 10% of probability mass uniformly over
    non-PAD, non-target tokens.
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  [N, vocab_size]
            targets: [N] long
        Returns:
            Scalar — summed loss (caller normalises by ntokens).
        """
        N = logits.size(0)
        log_probs = F.log_softmax(logits, dim=-1)  # [N, V]

        smooth_val = self.smoothing / (self.vocab_size - 2)  # exclude PAD + true class
        target_dist = torch.full_like(log_probs, smooth_val)
        target_dist[:, self.pad_id] = 0.0
        target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        # Zero out PAD-target rows entirely
        pad_mask = (targets == self.pad_id)
        target_dist[pad_mask] = 0.0

        return F.kl_div(log_probs, target_dist, reduction="sum")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    """Format a duration as mm:ss or Xm Ys."""
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _move_forward_batch(batch, device):
    batch.src_mat  = batch.src_mat.to(device)
    batch.src_thk  = batch.src_thk.to(device)
    batch.spectrum = batch.spectrum.to(device)
    batch.src_mask = batch.src_mask.to(device)
    return batch


def _move_inverse_batch(batch, device):
    batch.spectrum    = batch.spectrum.to(device)
    batch.tgt_mat     = batch.tgt_mat.to(device)
    batch.tgt_thk     = batch.tgt_thk.to(device)
    batch.tgt_y_mat   = batch.tgt_y_mat.to(device)
    batch.tgt_y_thk   = batch.tgt_y_thk.to(device)
    batch.tgt_mask    = batch.tgt_mask.to(device)
    return batch


# ── Training loops ─────────────────────────────────────────────────────────────

def train_forward(
    model,
    train_loader,
    dev_loader,
    optimizer: AdamW,
    scheduler,
    epochs: int,
    device: torch.device,
    save_dir: str,
    run_name: str,
    config: dict | None = None,
    vocab=None,
) -> list[dict]:
    """
    Train the ForwardModel.

    Checkpoints saved to save_dir/{run_name}/best.pt and latest.pt.
    Each checkpoint contains: model state dict, optimizer state, scheduler state,
    epoch, loss_history, config, vocab (word2id dict).

    Returns:
        loss_history — list of {"epoch", "train", "dev"} dicts
    """
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    criterion = nn.MSELoss()
    loss_history: list[dict] = []
    best_dev_loss = float("inf")

    vocab_dict = vocab.word2id if vocab is not None else {}
    n_train_batches = len(train_loader)
    n_dev_batches   = len(dev_loader)

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss_sum = 0.0
        n_train = 0
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_loader, 1):
            step_start = time.perf_counter()
            batch = _move_forward_batch(batch, device)
            optimizer.zero_grad()
            pred = model(batch.src_mat, batch.src_thk, batch.src_mask)
            loss = criterion(pred, batch.spectrum)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size     = batch.src_mat.size(0)
            train_loss_sum += loss.item() * batch_size
            n_train        += batch_size
            step_ms        = (time.perf_counter() - step_start) * 1000
            running_loss   = train_loss_sum / n_train
            current_lr     = scheduler.get_last_lr()[0]

            print(
                f"\rEpoch {epoch}/{epochs}  "
                f"step {step}/{n_train_batches}  "
                f"loss={running_loss:.6f}  "
                f"lr={current_lr:.2e}  "
                f"step={step_ms:.0f}ms",
                end="", flush=True,
            )

        train_loss  = train_loss_sum / max(n_train, 1)
        train_time  = time.perf_counter() - epoch_start

        # ── eval ──
        model.eval()
        dev_loss_sum = 0.0
        n_dev = 0
        eval_start = time.perf_counter()

        with torch.no_grad():
            for step, batch in enumerate(dev_loader, 1):
                batch = _move_forward_batch(batch, device)
                pred  = model(batch.src_mat, batch.src_thk, batch.src_mask)
                loss  = criterion(pred, batch.spectrum)
                dev_loss_sum += loss.item() * batch.src_mat.size(0)
                n_dev        += batch.src_mat.size(0)

                print(
                    f"\rEpoch {epoch}/{epochs}  "
                    f"eval {step}/{n_dev_batches}",
                    end="", flush=True,
                )

        dev_loss  = dev_loss_sum / max(n_dev, 1)
        eval_time = time.perf_counter() - eval_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"\rEpoch {epoch:4d}/{epochs}  "
            f"train={train_loss:.6f}  dev={dev_loss:.6f}  "
            f"train={_fmt(train_time)}  eval={_fmt(eval_time)}  "
            f"lr={current_lr:.2e}"
            + (" *" if dev_loss < best_dev_loss else "  ")
        )

        loss_history.append({"epoch": epoch, "train": train_loss, "dev": dev_loss})

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss_history": loss_history,
            "config": config or {},
            "vocab": vocab_dict,
        }
        torch.save(ckpt, os.path.join(run_dir, "latest.pt"))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    return loss_history


def train_inverse(
    model,
    train_loader,
    dev_loader,
    optimizer: AdamW,
    scheduler,
    epochs: int,
    device: torch.device,
    save_dir: str,
    run_name: str,
    vocab_size: int,
    pad_id: int = 0,
    config: dict | None = None,
    vocab=None,
) -> list[dict]:
    """
    Train the InverseModel.

    Loss = (label-smoothed KL for materials + masked MSE for thicknesses) / ntokens.

    Returns:
        loss_history — list of {"epoch", "train", "dev"} dicts
    """
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    mat_criterion = LabelSmoothingLoss(vocab_size, smoothing=0.1, pad_id=pad_id)
    loss_history: list[dict] = []
    best_dev_loss = float("inf")

    vocab_dict = vocab.word2id if vocab is not None else {}
    n_train_batches = len(train_loader)
    n_dev_batches   = len(dev_loader)

    def _forward_loss(batch) -> tuple[Tensor, int]:
        mat_logits, thk_pred = model(
            batch.spectrum, batch.tgt_mat, batch.tgt_thk, batch.tgt_mask
        )
        B, T, V = mat_logits.shape
        mat_loss = mat_criterion(mat_logits.view(B * T, V), batch.tgt_y_mat.view(-1))
        thk_mask = (batch.tgt_y_mat != pad_id).float()
        thk_loss = ((thk_pred - batch.tgt_y_thk) ** 2 * thk_mask).sum()
        ntokens  = batch.ntokens_tgt
        total    = (mat_loss + thk_loss) / max(ntokens, 1)
        return total, ntokens

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss_sum = 0.0
        train_tokens   = 0
        epoch_start    = time.perf_counter()

        for step, batch in enumerate(train_loader, 1):
            step_start = time.perf_counter()
            batch = _move_inverse_batch(batch, device)
            optimizer.zero_grad()
            loss, ntokens = _forward_loss(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item() * ntokens
            train_tokens   += ntokens
            step_ms         = (time.perf_counter() - step_start) * 1000
            running_loss    = train_loss_sum / max(train_tokens, 1)
            current_lr      = scheduler.get_last_lr()[0]

            print(
                f"\rEpoch {epoch}/{epochs}  "
                f"step {step}/{n_train_batches}  "
                f"loss={running_loss:.6f}  "
                f"lr={current_lr:.2e}  "
                f"step={step_ms:.0f}ms",
                end="", flush=True,
            )

        train_loss  = train_loss_sum / max(train_tokens, 1)
        train_time  = time.perf_counter() - epoch_start

        # ── eval ──
        model.eval()
        dev_loss_sum = 0.0
        dev_tokens   = 0
        eval_start   = time.perf_counter()

        with torch.no_grad():
            for step, batch in enumerate(dev_loader, 1):
                batch = _move_inverse_batch(batch, device)
                loss, ntokens = _forward_loss(batch)
                dev_loss_sum += loss.item() * ntokens
                dev_tokens   += ntokens

                print(
                    f"\rEpoch {epoch}/{epochs}  "
                    f"eval {step}/{n_dev_batches}",
                    end="", flush=True,
                )

        dev_loss  = dev_loss_sum / max(dev_tokens, 1)
        eval_time = time.perf_counter() - eval_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"\rEpoch {epoch:4d}/{epochs}  "
            f"train={train_loss:.6f}  dev={dev_loss:.6f}  "
            f"train={_fmt(train_time)}  eval={_fmt(eval_time)}  "
            f"lr={current_lr:.2e}"
            + (" *" if dev_loss < best_dev_loss else "  ")
        )

        loss_history.append({"epoch": epoch, "train": train_loss, "dev": dev_loss})

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss_history": loss_history,
            "config": config or {},
            "vocab": vocab_dict,
        }
        torch.save(ckpt, os.path.join(run_dir, "latest.pt"))
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(ckpt, os.path.join(run_dir, "best.pt"))

    return loss_history
