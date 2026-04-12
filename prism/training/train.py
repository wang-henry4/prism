"""
Training utilities: cosine-annealing LR schedule, LabelSmoothing loss, and
train_inverse loop with checkpoint saving.
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from prism.model.prefix_material_thk_model import InverseModel


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
    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=min_lr,
        )
        scheduler = SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=min_lr,
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


def _grad_stats(model: nn.Module) -> dict[str, float]:
    """Compute gradient statistics across all model parameters."""
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return {"grad_norm": 0.0, "grad_max": 0.0, "grad_zero_pct": 100.0}
    all_grads = torch.cat([g.flatten() for g in grads])
    total_norm = torch.norm(all_grads, 2).item()
    max_abs = all_grads.abs().max().item()
    zero_pct = (all_grads == 0).float().mean().item() * 100
    return {"grad_norm": total_norm, "grad_max": max_abs, "grad_zero_pct": zero_pct}


def _move_inverse_batch(batch, device):
    batch.spectrum    = batch.spectrum.to(device)
    batch.tgt_mat     = batch.tgt_mat.to(device)
    batch.tgt_thk     = batch.tgt_thk.to(device)
    batch.tgt_y_mat   = batch.tgt_y_mat.to(device)
    batch.tgt_y_thk   = batch.tgt_y_thk.to(device)
    batch.tgt_mask    = batch.tgt_mask.to(device)
    return batch


# ── Training loops ─────────────────────────────────────────────────────────────

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
    thk_loss_weight: float = 1.0,
    start_epoch: int = 1,
    prior_loss_history: list[dict] | None = None,
) -> list[dict]:
    """
    Train the InverseModel.

    Loss = (mat_loss + thk_loss_weight * thk_loss) / ntokens.

    Per-token mat_loss and thk_loss are logged each epoch so you can
    compare their scales and tune thk_loss_weight accordingly.

    Args:
        start_epoch:        Epoch number to start from (1-based). When resuming,
                            set to prior epoch + 1.
        prior_loss_history: Loss history from a prior checkpoint to prepend.

    Returns:
        loss_history — list of per-epoch dicts including component losses
    """
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    mat_criterion = LabelSmoothingLoss(vocab_size, smoothing=0.1, pad_id=pad_id)
    loss_history: list[dict] = list(prior_loss_history) if prior_loss_history else []
    best_dev_loss = min((h["dev"] for h in loss_history), default=float("inf"))

    vocab_dict = vocab.word2id if vocab is not None else {}
    n_train_batches = len(train_loader)
    n_dev_batches   = len(dev_loader)

    def _forward_loss(batch) -> tuple[Tensor, int, float, float]:
        """Returns (total_loss, ntokens, mat_loss_sum, thk_loss_sum)."""
        mat_logits, thk_pred = model(
            batch.spectrum, batch.tgt_mat, batch.tgt_thk, batch.tgt_mask
        )
        B, T, V = mat_logits.shape
        mat_loss = mat_criterion(mat_logits.view(B * T, V), batch.tgt_y_mat.view(-1))
        thk_mask = (batch.tgt_y_mat != pad_id).float()
        # Gather the thickness prediction at the ground-truth material index
        thk_pred = thk_pred.gather(-1, batch.tgt_y_mat.unsqueeze(-1)).squeeze(-1)
        # Compare in log-space
        thk_target = InverseModel.nm_to_log(batch.tgt_y_thk)
        thk_loss = ((thk_pred - thk_target) ** 2 * thk_mask).sum()
        ntokens  = batch.ntokens_tgt
        total    = (mat_loss + thk_loss_weight * thk_loss) / max(ntokens, 1)
        return total, ntokens, mat_loss.item(), thk_loss.item()

    end_epoch = start_epoch + epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        # ── train ──
        model.train()
        train_loss_sum = 0.0
        train_mat_sum  = 0.0
        train_thk_sum  = 0.0
        train_tokens   = 0
        epoch_start    = time.perf_counter()

        epoch_grad_norms: list[float] = []
        epoch_grad_maxs: list[float] = []

        for step, batch in enumerate(train_loader, 1):
            step_start = time.perf_counter()
            batch = _move_inverse_batch(batch, device)
            optimizer.zero_grad()
            loss, ntokens, mat_l, thk_l = _forward_loss(batch)
            loss.backward()
            gs = _grad_stats(model)
            epoch_grad_norms.append(gs["grad_norm"])
            epoch_grad_maxs.append(gs["grad_max"])
            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item() * ntokens
            train_mat_sum  += mat_l
            train_thk_sum  += thk_l
            train_tokens   += ntokens
            step_ms         = (time.perf_counter() - step_start) * 1000
            running_loss    = train_loss_sum / max(train_tokens, 1)
            current_lr      = scheduler.get_last_lr()[0]

            print(
                f"\rEpoch {epoch}/{end_epoch}  "
                f"step {step}/{n_train_batches}  "
                f"loss={running_loss:.6f}  "
                f"lr={current_lr:.8e}  "
                f"gnorm={gs['grad_norm']:.2f}  "
                f"gmax={gs['grad_max']:.2f}  "
                f"step={step_ms:.0f}ms",
                end="", flush=True,
            )

        train_loss  = train_loss_sum / max(train_tokens, 1)
        train_mat   = train_mat_sum  / max(train_tokens, 1)
        train_thk   = train_thk_sum  / max(train_tokens, 1)
        train_time  = time.perf_counter() - epoch_start

        # ── eval ──
        model.eval()
        dev_loss_sum = 0.0
        dev_mat_sum  = 0.0
        dev_thk_sum  = 0.0
        dev_tokens   = 0
        eval_start   = time.perf_counter()

        with torch.no_grad():
            for step, batch in enumerate(dev_loader, 1):
                batch = _move_inverse_batch(batch, device)
                loss, ntokens, mat_l, thk_l = _forward_loss(batch)
                dev_loss_sum += loss.item() * ntokens
                dev_mat_sum  += mat_l
                dev_thk_sum  += thk_l
                dev_tokens   += ntokens

                print(
                    f"\rEpoch {epoch}/{end_epoch}  "
                    f"eval {step}/{n_dev_batches}",
                    end="", flush=True,
                )

        dev_loss  = dev_loss_sum / max(dev_tokens, 1)
        dev_mat   = dev_mat_sum  / max(dev_tokens, 1)
        dev_thk   = dev_thk_sum  / max(dev_tokens, 1)
        eval_time = time.perf_counter() - eval_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"\rEpoch {epoch:4d}/{end_epoch}  "
            f"train={train_loss:.6f}  dev={dev_loss:.6f}  "
            f"mat={dev_mat:.4f}  thk={dev_thk:.4f}  "
            f"ratio={dev_thk / (dev_mat + 1e-10):.1f}  "
            f"train={_fmt(train_time)}  eval={_fmt(eval_time)}  "
            f"lr={current_lr:.8e}"
            + (" *" if dev_loss < best_dev_loss else "  ")
        )

        loss_history.append({
            "epoch": epoch,
            "train": train_loss,
            "dev": dev_loss,
            "train_mat": train_mat,
            "train_thk": train_thk,
            "dev_mat": dev_mat,
            "dev_thk": dev_thk,
            "thk_loss_weight": thk_loss_weight,
            "grad_norm_mean": sum(epoch_grad_norms) / len(epoch_grad_norms),
            "grad_norm_max": max(epoch_grad_norms),
            "grad_max_mean": sum(epoch_grad_maxs) / len(epoch_grad_maxs),
            "grad_max_max": max(epoch_grad_maxs),
        })

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
