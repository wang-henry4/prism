"""Spectrum evaluation metrics: MSE, MAE, R²."""

import numpy as np


class SpectrumMetrics:
    """Compute MSE, MAE, and R² for spectrum predictions."""

    @staticmethod
    def compute(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
        """
        Args:
            pred:   [N, 142] or [142] predicted spectra
            target: [N, 142] or [142] ground-truth spectra

        Returns:
            {"mse": ..., "mae": ..., "r2": ...}
        """
        mse = float(np.mean((pred - target) ** 2))
        mae = float(np.mean(np.abs(pred - target)))
        ss_res = np.sum((pred - target) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-10))
        return {"mse": mse, "mae": mae, "r2": r2}
