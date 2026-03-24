"""
Hand-crafted target spectra for qualitative evaluation.

Each target is a dict with:
  - name:     short identifier used in filenames
  - label:    human-readable description for plot titles
  - spectrum: list of 142 floats (71 R + 71 T) at 400-1100 nm, 10 nm steps

Add new targets by appending to HANDCRAFTED_TARGETS.
"""

import numpy as np

from optoformer.constants import N_WL, N_SPECTRUM, WL_NM


def _make_spectrum(reflectance: np.ndarray, transmittance: np.ndarray) -> list[float]:
    """Concatenate R and T into the 142-float spectrum format."""
    assert len(reflectance) == N_WL and len(transmittance) == N_WL
    return list(np.concatenate([reflectance, transmittance]))


def _shortpass_780() -> dict:
    """
    Shortpass filter with cutoff at 780 nm.
    R=0, T=1 for 400-770 nm; R=1, T=0 for 780-1100 nm.
    """
    r = np.zeros(N_WL)
    t = np.ones(N_WL)
    cutoff_idx = int((780 - WL_NM[0]) / (WL_NM[1] - WL_NM[0]))  # index 38
    r[cutoff_idx:] = 1.0
    t[cutoff_idx:] = 0.0
    return {
        "name": "shortpass_780",
        "label": "Shortpass 780 nm",
        "spectrum": _make_spectrum(r, t),
    }


# ── Registry ─────────────────────────────────────────────────────────────────

HANDCRAFTED_TARGETS: list[dict] = [
    _shortpass_780(),
]
