"""
TMM-based verifiable reward for RLVR.

Wraps the physics simulation into a batched reward computation that can
run in a multiprocessing pool for throughput.
"""

from multiprocessing import Pool

import numpy as np

from prism.constants import N_SPECTRUM
from prism.data.sim import load_nk, simulate


# ── Worker helpers (module-level for pickling) ────────────────────────────────

_nk_dict = None  # populated by _worker_init in each subprocess


def _worker_init(nk_dir: str) -> None:
    global _nk_dict
    _nk_dict = load_nk(nk_dir)


def _simulate_one(args: tuple[list[str], list[float]]) -> list[float]:
    materials, thicknesses = args
    if not materials:
        return [0.0] * N_SPECTRUM
    try:
        return simulate(materials, thicknesses, _nk_dict)  # type: ignore[arg-type]
    except Exception:
        return [0.0] * N_SPECTRUM


# ── Reward class ──────────────────────────────────────────────────────────────

class TMMReward:
    """Compute rewards by re-simulating predicted thin-film designs via TMM.

    Supports four reward modes:
        normalized_mse: 1 - MSE, bounded [0, 1] since spectra ∈ [0, 1]
        r2:             R² coefficient of determination (can be < 0 for flat spectra)
        neg_mse:        negative MSE (≤ 0, higher is better)
        neg_mae:        negative MAE (≤ 0, higher is better)
    """

    def __init__(
        self,
        nk_dir: str,
        n_workers: int = 8,
        reward_mode: str = "r2",
    ):
        self.nk_dir = nk_dir
        self.n_workers = n_workers
        self.reward_mode = reward_mode
        # Pre-load nk data for single-process fallback
        self.nk_dict = load_nk(nk_dir)

    def compute(
        self,
        mat_names_list: list[list[str]],
        thk_vals_list: list[list[float]],
        target_spectra: np.ndarray,         # [N, 142]
    ) -> np.ndarray:
        """Compute per-rollout rewards.

        Args:
            mat_names_list: N lists of material name strings.
            thk_vals_list:  N lists of thickness floats (nm).
            target_spectra: [N, 142] ground-truth or target spectra.

        Returns:
            [N] float array of reward values.
        """
        N = len(mat_names_list)
        assert len(thk_vals_list) == N
        assert target_spectra.shape == (N, N_SPECTRUM)

        # TMM re-simulation (parallelized)
        jobs = list(zip(mat_names_list, thk_vals_list))

        if self.n_workers > 1 and N > 1:
            with Pool(
                processes=min(self.n_workers, N),
                initializer=_worker_init,
                initargs=(self.nk_dir,),
            ) as pool:
                pred_spectra_list = pool.map(_simulate_one, jobs)
        else:
            # Single-process fallback
            pred_spectra_list = []
            for mats, thks in jobs:
                if not mats:
                    pred_spectra_list.append([0.0] * N_SPECTRUM)
                else:
                    try:
                        pred_spectra_list.append(
                            simulate(mats, thks, self.nk_dict)
                        )
                    except Exception:
                        pred_spectra_list.append([0.0] * N_SPECTRUM)

        pred_spectra = np.array(pred_spectra_list, dtype=np.float64)  # [N, 142]

        return self._score(pred_spectra, target_spectra)

    def _score(
        self,
        pred: np.ndarray,     # [N, 142]
        target: np.ndarray,   # [N, 142]
    ) -> np.ndarray:
        """Compute per-sample reward."""
        if self.reward_mode == "normalized_mse":
            # 1 - MSE: bounded [0, 1] since spectra ∈ [0, 1], worst-case MSE = 1.0
            return 1.0 - np.mean((pred - target) ** 2, axis=1)

        if self.reward_mode == "neg_mse":
            return -np.mean((pred - target) ** 2, axis=1)

        if self.reward_mode == "neg_mae":
            return -np.mean(np.abs(pred - target), axis=1)

        # Default: R²  (per-sample)
        ss_res = np.sum((pred - target) ** 2, axis=1)
        ss_tot = np.sum((target - target.mean(axis=1, keepdims=True)) ** 2, axis=1)
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        return r2
