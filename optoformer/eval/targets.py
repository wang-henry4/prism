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


def _wl_idx(wl_nm: float) -> int:
    """Convert a wavelength in nm to the nearest grid index."""
    return int(round((wl_nm - WL_NM[0]) / (WL_NM[1] - WL_NM[0])))


# ── Filter builders ──────────────────────────────────────────────────────────

def shortpass(cutoff_nm: float) -> dict:
    """Shortpass filter: transmits below cutoff, reflects above."""
    idx = _wl_idx(cutoff_nm)
    r = np.zeros(N_WL)
    t = np.ones(N_WL)
    r[idx:] = 1.0
    t[idx:] = 0.0
    return {
        "name": f"shortpass_{int(cutoff_nm)}",
        "label": f"Shortpass {int(cutoff_nm)} nm",
        "spectrum": _make_spectrum(r, t),
    }


def longpass(cutoff_nm: float) -> dict:
    """Longpass filter: reflects below cutoff, transmits above."""
    idx = _wl_idx(cutoff_nm)
    r = np.ones(N_WL)
    t = np.zeros(N_WL)
    r[idx:] = 0.0
    t[idx:] = 1.0
    return {
        "name": f"longpass_{int(cutoff_nm)}",
        "label": f"Longpass {int(cutoff_nm)} nm",
        "spectrum": _make_spectrum(r, t),
    }


def bandpass(lo_nm: float, hi_nm: float) -> dict:
    """Bandpass filter: transmits between lo and hi, reflects outside."""
    lo_idx = _wl_idx(lo_nm)
    hi_idx = _wl_idx(hi_nm)
    r = np.ones(N_WL)
    t = np.zeros(N_WL)
    r[lo_idx:hi_idx] = 0.0
    t[lo_idx:hi_idx] = 1.0
    return {
        "name": f"bandpass_{int(lo_nm)}_{int(hi_nm)}",
        "label": f"Bandpass {int(lo_nm)}–{int(hi_nm)} nm",
        "spectrum": _make_spectrum(r, t),
    }


def bandstop(lo_nm: float, hi_nm: float) -> dict:
    """Bandstop (notch) filter: reflects between lo and hi, transmits outside."""
    lo_idx = _wl_idx(lo_nm)
    hi_idx = _wl_idx(hi_nm)
    r = np.zeros(N_WL)
    t = np.ones(N_WL)
    r[lo_idx:hi_idx] = 1.0
    t[lo_idx:hi_idx] = 0.0
    return {
        "name": f"bandstop_{int(lo_nm)}_{int(hi_nm)}",
        "label": f"Bandstop {int(lo_nm)}–{int(hi_nm)} nm",
        "spectrum": _make_spectrum(r, t),
    }


# ── Registry ─────────────────────────────────────────────────────────────────

HANDCRAFTED_TARGETS: list[dict] = [
    shortpass(780),
    bandpass(600, 900),
]
