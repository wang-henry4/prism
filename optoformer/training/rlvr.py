"""
GRPO (Group Relative Policy Optimization) trainer for RLVR post-training.

Uses TMM re-simulation as a verifiable reward signal to fine-tune a
pretrained InverseModel beyond the proxy losses (material CE + thickness MSE).
"""

import math
import os
import time

import numpy as np
import pyarrow.feather as feather
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from optoformer.constants import N_SPECTRUM, THK_MIN, THK_MAX
from optoformer.data.dataset import Vocab
from optoformer.eval.decode import sample_decode, greedy_decode
from optoformer.eval.metrics import SpectrumMetrics
from optoformer.eval.targets import HANDCRAFTED_TARGETS
from optoformer.training.reward import TMMReward


# ── Data mixing ───────────────────────────────────────────────────────────────

class MixedSpectrumSource:
    """Yields batches of target spectra mixed from Arrow data and handcrafted targets.

    Each batch contains:
        - (1 - handcrafted_ratio) * batch_size spectra from Arrow files
        - handcrafted_ratio * batch_size spectra from handcrafted targets

    Iterates over Arrow data in shuffled order; handcrafted targets are sampled
    with replacement each batch.
    """

    def __init__(
        self,
        arrow_paths: list[str],
        handcrafted_targets: list[dict],
        batch_size: int = 32,
        handcrafted_ratio: float = 0.2,
    ):
        self.batch_size = batch_size
        self.handcrafted_ratio = handcrafted_ratio

        # Load all spectra from Arrow files into memory
        spectra_list = []
        for path in arrow_paths:
            table = feather.read_table(path, memory_map=True)
            spectra_list.extend(table["spectra"].to_pylist())
        self.arrow_spectra = np.array(spectra_list, dtype=np.float32)  # [N_arrow, 142]
        self.n_arrow = len(self.arrow_spectra)

        # Handcrafted spectra
        if handcrafted_targets:
            self.hc_spectra = np.array(
                [t["spectrum"] for t in handcrafted_targets], dtype=np.float32
            )  # [N_hc, 142]
        else:
            self.hc_spectra = np.zeros((0, N_SPECTRUM), dtype=np.float32)
        self.n_hc = len(self.hc_spectra)

        # Shuffled index for Arrow data
        self._arrow_idx = np.random.permutation(self.n_arrow)
        self._arrow_pos = 0

    def _next_arrow_batch(self, n: int) -> np.ndarray:
        """Get n spectra from Arrow data, reshuffling when exhausted."""
        if self._arrow_pos + n > self.n_arrow:
            self._arrow_idx = np.random.permutation(self.n_arrow)
            self._arrow_pos = 0
        indices = self._arrow_idx[self._arrow_pos : self._arrow_pos + n]
        self._arrow_pos += n
        return self.arrow_spectra[indices]

    def __iter__(self):
        return self

    def __next__(self) -> Tensor:
        n_hc = 0
        if self.n_hc > 0:
            n_hc = int(self.batch_size * self.handcrafted_ratio)
        n_arrow = self.batch_size - n_hc

        parts = []
        if n_arrow > 0:
            parts.append(self._next_arrow_batch(n_arrow))
        if n_hc > 0:
            hc_indices = np.random.randint(0, self.n_hc, size=n_hc)
            parts.append(self.hc_spectra[hc_indices])

        batch = np.concatenate(parts, axis=0)
        # Shuffle within batch so handcrafted aren't always at the end
        perm = np.random.permutation(len(batch))
        return torch.tensor(batch[perm], dtype=torch.float32)


# ── Reference policy log-probs ────────────────────────────────────────────────

def compute_ref_log_probs(
    ref_model: nn.Module,
    mat_ids_list: list[list[int]],
    thk_vals_list: list[list[float]],
    spectra: Tensor,          # [N, 142]
    vocab: Vocab,
    thk_noise_std: float,
    device: torch.device,
) -> Tensor:
    """Teacher-force rollouts through the frozen reference model.

    Returns:
        total_log_probs: [N] — summed material + thickness log-probs per rollout.
    """
    N = len(mat_ids_list)
    log_norm_const = math.log(thk_noise_std * math.sqrt(2.0 * math.pi))

    total_lps = torch.zeros(N, device=device)

    with torch.no_grad():
        for i in range(N):
            mats = mat_ids_list[i]
            thks = thk_vals_list[i]
            if not mats:
                continue

            # Build teacher-forced input: [BOS, mat_1, ..., mat_L]
            input_ids = [vocab.BOS] + mats
            input_thk = [0.0] + thks
            # Target: [mat_1, ..., mat_L, EOS]
            target_ids = mats + [vocab.EOS]
            target_thk = thks + [0.0]

            mat_seq = torch.tensor([input_ids], dtype=torch.long, device=device)    # [1, T]
            thk_seq = torch.tensor([input_thk], dtype=torch.float32, device=device) # [1, T]
            spec = spectra[i:i+1].to(device)                                         # [1, 142]

            T = mat_seq.size(1)
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
            tgt_mask = causal.unsqueeze(0)

            mat_logits, thk_pred = ref_model(spec, mat_seq, thk_seq, tgt_mask)
            if hasattr(ref_model, "thk_to_nm"):
                thk_pred = ref_model.thk_to_nm(thk_pred)
            # mat_logits: [1, T, V], thk_pred: [1, T, V] (nm)

            log_probs = F.log_softmax(mat_logits[0], dim=-1)  # [T, V]
            target_t = torch.tensor(target_ids, dtype=torch.long, device=device)

            # Material log-probs
            mat_lp = log_probs.gather(-1, target_t.unsqueeze(-1)).squeeze(-1)  # [T]
            # Exclude EOS position for thickness (thickness=0 at EOS)
            mat_lp_sum = mat_lp.sum()

            # Thickness log-probs (Gaussian)
            thk_lp_sum = 0.0
            per_material_thk = thk_pred.dim() == 3
            for t_idx in range(len(mats)):
                if per_material_thk:
                    mu = thk_pred[0, t_idx, target_ids[t_idx]]
                else:
                    mu = thk_pred[0, t_idx]
                thk_val = target_thk[t_idx]
                lp = -0.5 * ((thk_val - mu) / thk_noise_std) ** 2 - log_norm_const
                thk_lp_sum = thk_lp_sum + lp

            total_lps[i] = mat_lp_sum + thk_lp_sum

    return total_lps


# ── Batched reference log-probs (more efficient) ─────────────────────────────

def compute_ref_log_probs_batched(
    ref_model: nn.Module,
    mat_ids_list: list[list[int]],
    thk_vals_list: list[list[float]],
    spectra: Tensor,          # [N, 142]
    vocab: Vocab,
    thk_noise_std: float,
    device: torch.device,
) -> Tensor:
    """Batched version — pads all rollouts and runs a single forward pass."""
    N = len(mat_ids_list)
    log_norm_const = math.log(thk_noise_std * math.sqrt(2.0 * math.pi))

    # Build padded teacher-forcing tensors
    max_input_len = max(len(m) + 1 for m in mat_ids_list)  # +1 for BOS

    input_mat = torch.full((N, max_input_len), vocab.PAD, dtype=torch.long, device=device)
    input_thk = torch.zeros(N, max_input_len, device=device)
    target_mat = torch.full((N, max_input_len), vocab.PAD, dtype=torch.long, device=device)
    target_thk = torch.zeros(N, max_input_len, device=device)
    seq_lens = torch.zeros(N, dtype=torch.long, device=device)

    for i in range(N):
        mats = mat_ids_list[i]
        thks = thk_vals_list[i]
        L = len(mats)
        seq_lens[i] = L + 1  # includes EOS in target

        # Input: [BOS, mat_1, ..., mat_L]
        input_mat[i, 0] = vocab.BOS
        for j, m in enumerate(mats):
            input_mat[i, j + 1] = m
            input_thk[i, j + 1] = thks[j]

        # Target: [mat_1, ..., mat_L, EOS]
        for j, m in enumerate(mats):
            target_mat[i, j] = m
            target_thk[i, j] = thks[j]
        target_mat[i, L] = vocab.EOS

    # Build causal mask
    T = max_input_len
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    pad_mask = (input_mat != vocab.PAD).unsqueeze(1)  # [N, 1, T]
    tgt_mask = pad_mask & causal.unsqueeze(0)          # [N, T, T]

    spec = spectra.to(device)

    with torch.no_grad():
        mat_logits, thk_pred = ref_model(spec, input_mat, input_thk, tgt_mask)
        if hasattr(ref_model, "thk_to_nm"):
            thk_pred = ref_model.thk_to_nm(thk_pred)

    log_probs = F.log_softmax(mat_logits, dim=-1)  # [N, T, V]

    # Gather material log-probs at target tokens
    mat_lp = log_probs.gather(-1, target_mat.unsqueeze(-1)).squeeze(-1)  # [N, T]
    # Mask out padding positions
    valid_mask = (target_mat != vocab.PAD).float()  # [N, T]
    mat_lp_sum = (mat_lp * valid_mask).sum(dim=1)  # [N]

    # Thickness log-probs (only at non-EOS, non-PAD target positions)
    per_material_thk = thk_pred.dim() == 3
    if per_material_thk:
        thk_mu = thk_pred.gather(-1, target_mat.unsqueeze(-1)).squeeze(-1)  # [N, T]
    else:
        thk_mu = thk_pred  # [N, T]

    thk_lp = -0.5 * ((target_thk - thk_mu) / thk_noise_std) ** 2 - log_norm_const
    # Only count thickness log-prob for actual material positions (not EOS/PAD)
    thk_valid = valid_mask.clone()
    for i in range(N):
        L = len(mat_ids_list[i])
        # Zero out EOS position (index L) for thickness
        if L < max_input_len:
            thk_valid[i, L] = 0.0
    thk_lp_sum = (thk_lp * thk_valid).sum(dim=1)  # [N]

    return mat_lp_sum + thk_lp_sum


# ── Group-relative advantage ─────────────────────────────────────────────────

def group_normalize(
    rewards: Tensor,      # [N]
    group_ids: Tensor,    # [N] long — which group each sample belongs to
    eps: float = 1e-8,
) -> Tensor:
    """Compute (reward - group_mean) / group_std for GRPO advantages."""
    unique_groups = group_ids.unique()
    advantages = torch.zeros_like(rewards)

    for g in unique_groups:
        mask = group_ids == g
        group_rewards = rewards[mask]
        mean = group_rewards.mean()
        std = group_rewards.std()
        advantages[mask] = (group_rewards - mean) / (std + eps)

    return advantages


# ── Logging helpers ───────────────────────────────────────────────────────────

def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ── Main GRPO training loop ──────────────────────────────────────────────────

def train_rlvr(
    model: nn.Module,
    ref_model: nn.Module,
    reward_fn: TMMReward,
    data_source: MixedSpectrumSource,
    vocab: Vocab,
    *,
    # GRPO hyperparams
    n_samples: int = 8,
    temperature: float = 1.0,
    thk_noise_std: float = 5.0,
    kl_coeff: float = 0.1,
    max_grad_norm: float = 1.0,
    # Optimizer
    lr: float = 1e-5,
    total_steps: int = 5000,
    warmup_steps: int = 100,
    # Eval & checkpointing
    eval_every: int = 100,
    eval_spectra: Tensor | None = None,       # [N_eval, 142] dev spectra for periodic eval
    save_dir: str = "saved_models/rlvr",
    run_name: str = "rlvr_v1",
    device: torch.device = torch.device("cpu"),
    config: dict | None = None,
) -> list[dict]:
    """
    GRPO training loop.

    For each step:
        1. Sample a batch of target spectra
        2. Generate G stochastic rollouts per spectrum
        3. Re-simulate via TMM → verifiable rewards
        4. Compute group-relative advantages
        5. Policy gradient + KL penalty update

    Returns:
        step_history: list of per-step metric dicts
    """
    run_dir = os.path.join(save_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model.train()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    step_history: list[dict] = []
    best_eval_r2 = -float("inf")
    vocab_dict = vocab.word2id

    print(f"RLVR training: {total_steps} steps, G={n_samples}, "
          f"T={temperature}, σ_thk={thk_noise_std}, KL={kl_coeff}")

    for step in range(1, total_steps + 1):
        step_start = time.perf_counter()

        # ── 1. Get batch of target spectra ──
        spectra_batch = next(data_source).to(device)  # [B, 142]
        B = spectra_batch.size(0)

        # ── 2. Sample G rollouts per spectrum ──
        model.train()
        rollouts = sample_decode(
            model, spectra_batch, vocab,
            n_samples=n_samples,
            temperature=temperature,
            thk_noise_std=thk_noise_std,
            device=device,
        )

        N = B * n_samples
        group_ids = rollouts["group_ids"]

        # ── 3. Compute TMM rewards (detached, on CPU) ──
        mat_names_list = []
        for mat_ids in rollouts["mat_ids"]:
            mat_names_list.append([vocab.decode(m) for m in mat_ids])
        thk_vals_list = rollouts["thk_vals"]

        # Expand target spectra to match rollouts: each spectrum repeated G times
        target_np = spectra_batch.cpu().numpy()                    # [B, 142]
        target_expanded = np.repeat(target_np, n_samples, axis=0)  # [N, 142]

        rewards_np = reward_fn.compute(mat_names_list, thk_vals_list, target_expanded)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)

        # ── 4. Group-relative advantages ──
        advantages = group_normalize(rewards, group_ids)

        # ── 5. Policy gradient loss ──
        pg_loss = -(advantages.detach() * rollouts["total_log_probs"]).mean()

        # ── 6. KL penalty against reference ──
        ref_spectra = spectra_batch.unsqueeze(1).expand(B, n_samples, -1).reshape(N, -1)
        ref_lps = compute_ref_log_probs_batched(
            ref_model,
            rollouts["mat_ids"],
            rollouts["thk_vals"],
            ref_spectra,
            vocab,
            thk_noise_std,
            device,
        )
        kl = (rollouts["total_log_probs"] - ref_lps).mean()

        # ── 7. Total loss and update ──
        loss = pg_loss + kl_coeff * kl

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        step_ms = (time.perf_counter() - step_start) * 1000

        # ── Logging ──
        mean_reward = rewards.mean().item()
        std_reward = rewards.std().item()
        # Within-group std: how much reward varies among G rollouts per spectrum
        rewards_grouped = rewards.view(B, n_samples)
        intra_group_std = rewards_grouped.std(dim=1).mean().item()
        mean_length = rollouts["lengths"].float().mean().item()
        current_lr = scheduler.get_last_lr()[0]

        step_info = {
            "step": step,
            "loss": loss.item(),
            "pg_loss": pg_loss.item(),
            "kl": kl.item(),
            "reward_mean": mean_reward,
            "reward_std": std_reward,
            "reward_intra_group_std": intra_group_std,
            "mean_length": mean_length,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm,
            "lr": current_lr,
        }
        step_history.append(step_info)

        print(
            f"\rStep {step:5d}/{total_steps}  "
            f"loss={loss.item():.4f}  "
            f"pg={pg_loss.item():.4f}  "
            f"kl={kl.item():.4f}  "
            f"R={mean_reward:.4f}±{std_reward:.3f}  "
            f"intra_σ={intra_group_std:.3f}  "
            f"len={mean_length:.1f}  "
            f"gnorm={step_info['grad_norm']:.2f}  "
            f"lr={current_lr:.2e}  "
            f"{step_ms:.0f}ms",
            end="", flush=True,
        )

        # ── Periodic eval ──
        if step % eval_every == 0 or step == total_steps:
            print()  # newline after progress
            eval_metrics = _evaluate_rlvr(model, vocab, eval_spectra, reward_fn, device)
            step_info.update(eval_metrics)

            print(
                f"  [eval] dev_R²={eval_metrics.get('eval_r2', 0):.4f}  "
                f"hc_R²={eval_metrics.get('hc_r2_mean', 0):.4f}  "
            )

            # Checkpoint
            ckpt = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step_history": step_history,
                "config": config or {},
                "vocab": vocab_dict,
                "rlvr_config": {
                    "n_samples": n_samples,
                    "temperature": temperature,
                    "thk_noise_std": thk_noise_std,
                    "kl_coeff": kl_coeff,
                    "lr": lr,
                },
            }
            torch.save(ckpt, os.path.join(run_dir, "latest.pt"))

            eval_r2 = eval_metrics.get("eval_r2", -float("inf"))
            if eval_r2 > best_eval_r2:
                best_eval_r2 = eval_r2
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
                print(f"  [eval] new best R²={eval_r2:.4f} *")

    print(f"\nRLVR training complete. Best eval R²={best_eval_r2:.4f}")
    return step_history


# ── Eval helper ───────────────────────────────────────────────────────────────

def _evaluate_rlvr(
    model: nn.Module,
    vocab: Vocab,
    eval_spectra: Tensor | None,
    reward_fn: TMMReward,
    device: torch.device,
    max_eval: int = 200,
) -> dict:
    """Quick evaluation via greedy decode + TMM re-sim."""
    model.eval()
    metrics = {}

    # Dev spectra
    if eval_spectra is not None and len(eval_spectra) > 0:
        spec = eval_spectra[:max_eval].to(device)
        with torch.no_grad():
            mat_ids, thk_vals = greedy_decode(model, spec, vocab, device=device)

        mat_names = [[vocab.decode(m) for m in ids] for ids in mat_ids]
        target_np = spec.cpu().numpy()

        rewards = reward_fn.compute(mat_names, thk_vals, target_np)
        metrics["eval_r2"] = float(np.mean(rewards))
        metrics["eval_r2_std"] = float(np.std(rewards))

    # Handcrafted targets
    if HANDCRAFTED_TARGETS:
        hc_spec = torch.tensor(
            [t["spectrum"] for t in HANDCRAFTED_TARGETS], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            mat_ids, thk_vals = greedy_decode(model, hc_spec, vocab, device=device)

        mat_names = [[vocab.decode(m) for m in ids] for ids in mat_ids]
        target_np = hc_spec.cpu().numpy()

        rewards = reward_fn.compute(mat_names, thk_vals, target_np)
        metrics["hc_r2_mean"] = float(np.mean(rewards))
        metrics["hc_r2_std"] = float(np.std(rewards))
        metrics["hc_r2_per_target"] = {
            HANDCRAFTED_TARGETS[i]["name"]: float(rewards[i])
            for i in range(len(HANDCRAFTED_TARGETS))
        }

    model.train()
    return metrics
