"""
Optimization-based inverse design baselines.

1. Differentiable TMM (gradient-based with L-BFGS, multi-start)
2. Simulated Annealing (global stochastic search)
3. Needle Optimization (iterative layer insertion + local optimization)

All methods: target spectrum → optimized (materials, thicknesses) → TMM re-sim → metrics.
"""

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from optoformer.constants import (
    MATERIALS, N_WL, THK_MAX, THK_MIN, THK_STEP, MAX_LAYERS,
)
from baselines.diff_tmm import _build_nk_tensor, tmm_spectrum


# ── Shared utilities ──────────────────────────────────────────────────────────

@dataclass
class DesignResult:
    materials: list[str]
    thicknesses: list[float]
    mae: float


def _spectrum_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


# ── 1. Differentiable TMM ────────────────────────────────────────────────────

def diff_tmm_optimize(
    target_spectrum: np.ndarray,
    mat_nk: Tensor,
    sub_nk: Tensor,
    n_starts: int = 32,
    n_layers_list: list[int] | None = None,
    max_iter: int = 300,
    device: str = "cuda",
) -> DesignResult:
    """
    Gradient-based inverse design via differentiable TMM.

    For each start: pick random materials and layer count, optimize thicknesses
    with L-BFGS. Keep the best result across all starts.

    Args:
        target_spectrum: [142] target R+T spectrum.
        mat_nk:  [N_MAT, N_WL] complex tensor of material refractive indices.
        sub_nk:  [N_WL] complex tensor of substrate refractive index.
        n_starts: Number of random restarts.
        n_layers_list: Layer counts to try. Default: [3,5,7,10,14,18].
        max_iter: L-BFGS iterations per start.
        device: torch device.
    """
    if n_layers_list is None:
        n_layers_list = [3, 5, 7, 10, 14, 18]

    target_t = torch.tensor(target_spectrum, dtype=torch.float32, device=device)
    mat_nk_d = mat_nk.to(device)
    sub_nk_d = sub_nk.to(device)
    n_mat = len(MATERIALS)

    best = DesignResult([], [], float("inf"))

    for _ in range(n_starts):
        n_layers = random.choice(n_layers_list)
        mat_indices = [random.randrange(n_mat) for _ in range(n_layers)]
        nk_layers = mat_nk_d[mat_indices]  # [n_layers, N_WL]

        # Initialize thicknesses randomly in valid range
        thk_raw = torch.empty(n_layers, device=device).uniform_(
            math.log(THK_MIN), math.log(THK_MAX)
        )
        thk_raw.requires_grad_(True)

        optimizer = torch.optim.LBFGS([thk_raw], max_iter=20, line_search_fn="strong_wolfe")

        for _ in range(max_iter // 20):
            def closure():
                optimizer.zero_grad()
                thk = thk_raw.exp().clamp(THK_MIN, THK_MAX)
                pred = tmm_spectrum(thk, nk_layers, sub_nk_d)
                loss = ((pred - target_t) ** 2).mean()
                loss.backward()
                return loss

            try:
                optimizer.step(closure)
            except (RuntimeError, IndexError):
                break

        with torch.no_grad():
            thk = thk_raw.exp().clamp(THK_MIN, THK_MAX)
            pred = tmm_spectrum(thk, nk_layers, sub_nk_d).cpu().numpy()
            mae = _spectrum_mae(pred, target_spectrum)

        if mae < best.mae:
            best = DesignResult(
                materials=[MATERIALS[i] for i in mat_indices],
                thicknesses=thk.cpu().tolist(),
                mae=mae,
            )

    return best


# ── 2. Simulated Annealing ───────────────────────────────────────────────────

def _sa_evaluate(
    mat_indices: list[int],
    thicknesses: list[float],
    target_spectrum: np.ndarray,
    mat_nk: Tensor,
    sub_nk: Tensor,
    device: str,
) -> float:
    """Evaluate a design candidate via differentiable TMM (no grad)."""
    with torch.no_grad():
        nk = mat_nk[mat_indices].to(device)
        thk = torch.tensor(thicknesses, dtype=torch.float32, device=device)
        pred = tmm_spectrum(thk, nk, sub_nk.to(device)).cpu().numpy()
    return _spectrum_mae(pred, target_spectrum)


def simulated_annealing(
    target_spectrum: np.ndarray,
    mat_nk: Tensor,
    sub_nk: Tensor,
    n_restarts: int = 8,
    n_iter: int = 5000,
    T_start: float = 0.1,
    T_end: float = 1e-4,
    device: str = "cuda",
) -> DesignResult:
    """
    Simulated annealing over (materials, thicknesses) space.

    Moves: thickness perturbation, material swap, layer add/remove.
    """
    n_mat = len(MATERIALS)
    best = DesignResult([], [], float("inf"))

    for _ in range(n_restarts):
        # Random initial design
        n_layers = random.randint(2, MAX_LAYERS)
        cur_mats = [random.randrange(n_mat) for _ in range(n_layers)]
        cur_thks = [random.uniform(THK_MIN, THK_MAX) for _ in range(n_layers)]
        cur_mae = _sa_evaluate(cur_mats, cur_thks, target_spectrum, mat_nk, sub_nk, device)

        local_best_mae = cur_mae
        local_best_mats = list(cur_mats)
        local_best_thks = list(cur_thks)

        for step in range(n_iter):
            T = T_start * (T_end / T_start) ** (step / n_iter)

            # Propose a move
            new_mats = list(cur_mats)
            new_thks = list(cur_thks)
            move = random.random()

            if move < 0.5:
                # Perturb a thickness
                idx = random.randrange(len(new_thks))
                new_thks[idx] = max(THK_MIN, min(THK_MAX,
                    new_thks[idx] + random.gauss(0, 30)))
            elif move < 0.7:
                # Swap a material
                idx = random.randrange(len(new_mats))
                new_mats[idx] = random.randrange(n_mat)
            elif move < 0.85 and len(new_mats) < MAX_LAYERS:
                # Add a layer
                pos = random.randrange(len(new_mats) + 1)
                new_mats.insert(pos, random.randrange(n_mat))
                new_thks.insert(pos, random.uniform(THK_MIN, THK_MAX))
            elif len(new_mats) > 1:
                # Remove a layer
                pos = random.randrange(len(new_mats))
                new_mats.pop(pos)
                new_thks.pop(pos)

            new_mae = _sa_evaluate(new_mats, new_thks, target_spectrum, mat_nk, sub_nk, device)
            delta = new_mae - cur_mae

            if delta < 0 or random.random() < math.exp(-delta / T):
                cur_mats, cur_thks, cur_mae = new_mats, new_thks, new_mae

            if cur_mae < local_best_mae:
                local_best_mae = cur_mae
                local_best_mats = list(cur_mats)
                local_best_thks = list(cur_thks)

        if local_best_mae < best.mae:
            best = DesignResult(
                materials=[MATERIALS[i] for i in local_best_mats],
                thicknesses=local_best_thks,
                mae=local_best_mae,
            )

    return best


# ── 3. Needle Optimization ───────────────────────────────────────────────────

def needle_optimization(
    target_spectrum: np.ndarray,
    mat_nk: Tensor,
    sub_nk: Tensor,
    max_layers: int = MAX_LAYERS,
    max_iter_per_insert: int = 200,
    device: str = "cuda",
) -> DesignResult:
    """
    Needle optimization: iteratively insert layers where they help most,
    then locally optimize thicknesses with L-BFGS.

    1. Start with a single layer (try all materials, keep best).
    2. For each insertion step:
       a. Try inserting a thin "needle" layer (each material) at each position.
       b. Keep the insertion that reduces MAE the most.
       c. Optimize all thicknesses with L-BFGS.
    3. Stop when no insertion improves the design or max_layers reached.
    """
    target_t = torch.tensor(target_spectrum, dtype=torch.float32, device=device)
    mat_nk_d = mat_nk.to(device)
    sub_nk_d = sub_nk.to(device)
    n_mat = len(MATERIALS)
    needle_thk = 20.0  # initial thickness for inserted needle

    def _optimize_thicknesses(mat_indices, init_thks):
        """L-BFGS optimization of thicknesses for fixed materials."""
        nk = mat_nk_d[mat_indices]
        thk_raw = torch.log(torch.tensor(init_thks, dtype=torch.float32, device=device))
        thk_raw.requires_grad_(True)
        opt = torch.optim.LBFGS([thk_raw], max_iter=20, line_search_fn="strong_wolfe")

        for _ in range(max_iter_per_insert // 20):
            def closure():
                opt.zero_grad()
                thk = thk_raw.exp().clamp(THK_MIN, THK_MAX)
                pred = tmm_spectrum(thk, nk, sub_nk_d)
                loss = ((pred - target_t) ** 2).mean()
                loss.backward()
                return loss
            try:
                opt.step(closure)
            except (RuntimeError, IndexError):
                break

        with torch.no_grad():
            thk = thk_raw.exp().clamp(THK_MIN, THK_MAX)
            pred = tmm_spectrum(thk, nk, sub_nk_d).cpu().numpy()
        return thk.cpu().tolist(), _spectrum_mae(pred, target_spectrum)

    # Step 1: find best single-layer starting point
    best_mats, best_thks, best_mae = None, None, float("inf")
    for m in range(n_mat):
        thks, mae = _optimize_thicknesses([m], [needle_thk])
        if mae < best_mae:
            best_mats, best_thks, best_mae = [m], thks, mae

    # Step 2: iteratively insert needles
    for _ in range(max_layers - 1):
        n_cur = len(best_mats)
        insert_best_mae = best_mae
        insert_best_mats = None
        insert_best_thks = None

        for pos in range(n_cur + 1):
            for m in range(n_mat):
                trial_mats = list(best_mats)
                trial_thks = list(best_thks)
                trial_mats.insert(pos, m)
                trial_thks.insert(pos, needle_thk)

                thks, mae = _optimize_thicknesses(trial_mats, trial_thks)
                if mae < insert_best_mae:
                    insert_best_mae = mae
                    insert_best_mats = list(trial_mats)
                    insert_best_thks = thks

        if insert_best_mats is None:
            break  # no insertion improved the design

        best_mats = insert_best_mats
        best_thks = insert_best_thks
        best_mae = insert_best_mae

    return DesignResult(
        materials=[MATERIALS[i] for i in best_mats],
        thicknesses=best_thks,
        mae=best_mae,
    )
