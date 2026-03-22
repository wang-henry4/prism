"""
TMM simulation and nk data loading for thin-film optics.

Wavelength grid: 400–1100 nm in 10 nm steps (71 points).
Spectrum output: 71 reflectance values + 71 transmittance values (142 floats total).
"""

import os

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from tmm import inc_tmm

from optoformer.constants import MATERIALS, SUBSTRATE, WL_NM, WL_UM, N_WL


def load_nk(nk_dir: str, materials: list[str] | None = None) -> dict[str, np.ndarray]:
    """
    Load refractive-index (n + ik) data for all materials and interpolate to WL_NM.

    Each CSV has columns: nm, n, k, wl  (wl in µm).
    Cubic-spline interpolation is used for both n and k; k is clipped to ≥ 0.

    Args:
        nk_dir:    Directory containing per-material CSV files.
        materials: Material names to load. Defaults to MATERIALS + [SUBSTRATE].

    Returns:
        Dict mapping material name → complex ndarray of shape (71,).
    """
    if materials is None:
        materials = MATERIALS + [SUBSTRATE]

    nk_dict: dict[str, np.ndarray] = {}
    for mat in materials:
        df = pd.read_csv(os.path.join(nk_dir, f"{mat}.csv"))
        df.dropna(inplace=True)
        df.sort_values("wl", inplace=True)

        wl = df["wl"].to_numpy()   # µm
        n  = df["n"].to_numpy()
        k  = df["k"].to_numpy()

        n_interp = CubicSpline(wl, n, extrapolate=True)(WL_UM)
        k_interp = CubicSpline(wl, k, extrapolate=True)(WL_UM).clip(min=0.0)

        nk_dict[mat] = n_interp + 1j * k_interp

    return nk_dict


def simulate(
    materials: list[str],
    thicknesses: list[float],
    nk_dict: dict[str, np.ndarray],
    substrate: str = SUBSTRATE,
    pol: str = "s",
    theta: float = 0.0,
) -> list[float]:
    """
    Simulate reflectance and transmittance of a thin-film stack via TMM.

    The stack is:  air (semi-inf) | layers… | substrate (500 µm) | air (semi-inf).
    Thin-film layers are treated as coherent; air and substrate as incoherent.

    Args:
        materials:   Layer material names (thin-film layers only, no substrate).
        thicknesses: Layer thicknesses in nm, same length as materials.
        nk_dict:     Pre-loaded refractive index dict from load_nk().
        substrate:   Substrate material name.
        pol:         Polarisation ('s' or 'p').
        theta:       Angle of incidence in degrees.

    Returns:
        142 floats — 71 R values followed by 71 T values (400–1100 nm).
    """
    theta_rad = np.deg2rad(theta)
    n_layers = len(materials)

    # inc_tmm layer classification: incoherent for air and substrate
    inc_list = ["i"] + ["c"] * n_layers + ["i", "i"]
    # Thickness list in nm: inf for air, layer thicknesses, 500 µm substrate, inf
    d_list = [np.inf] + list(thicknesses) + [500_000.0, np.inf]

    R, T = [], []
    for i, wl_nm in enumerate(WL_NM):
        n_list = (
            [1.0]
            + [nk_dict[m][i] for m in materials]
            + [nk_dict[substrate][i], 1.0]
        )
        res = inc_tmm(pol, n_list, d_list, inc_list, theta_rad, wl_nm)
        R.append(float(res["R"]))
        T.append(float(res["T"]))

    return R + T
