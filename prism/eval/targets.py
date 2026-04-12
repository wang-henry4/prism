"""
Hand-crafted target spectra for qualitative evaluation.

Each target is a dict with:
  - name:     short identifier used in filenames
  - label:    human-readable description for plot titles
  - spectrum: list of 142 floats (71 R + 71 T) at 400-1100 nm, 10 nm steps

Add new targets by appending to HANDCRAFTED_TARGETS.
"""

import numpy as np

from prism.constants import N_WL, N_SPECTRUM, WL_NM


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


def narrowband(center_nm: float, fwhm_nm: float = 10.0) -> dict:
    """Narrowband bandpass filter (FWHM < 30 nm) with Gaussian profile."""
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t = np.exp(-0.5 * ((WL_NM - center_nm) / sigma) ** 2)
    r = 1.0 - t
    return {
        "name": f"narrowband_{int(center_nm)}_fwhm{int(fwhm_nm)}",
        "label": f"Narrowband {int(center_nm)} nm (FWHM {int(fwhm_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def broadband(center_nm: float, fwhm_nm: float = 100.0) -> dict:
    """Broadband bandpass filter (FWHM > 50 nm) with Gaussian profile."""
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t = np.exp(-0.5 * ((WL_NM - center_nm) / sigma) ** 2)
    r = 1.0 - t
    return {
        "name": f"broadband_{int(center_nm)}_fwhm{int(fwhm_nm)}",
        "label": f"Broadband {int(center_nm)} nm (FWHM {int(fwhm_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def steep_longpass(cutoff_nm: float, steepness_nm: float = 10.0) -> dict:
    """Longpass edge filter with steep sigmoid transition."""
    t = 1.0 / (1.0 + np.exp(-(WL_NM - cutoff_nm) / (steepness_nm / 4.0)))
    r = 1.0 - t
    return {
        "name": f"steep_longpass_{int(cutoff_nm)}_s{int(steepness_nm)}",
        "label": f"Steep Longpass {int(cutoff_nm)} nm (edge {int(steepness_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def steep_shortpass(cutoff_nm: float, steepness_nm: float = 10.0) -> dict:
    """Shortpass edge filter with steep sigmoid transition."""
    t = 1.0 / (1.0 + np.exp((WL_NM - cutoff_nm) / (steepness_nm / 4.0)))
    r = 1.0 - t
    return {
        "name": f"steep_shortpass_{int(cutoff_nm)}_s{int(steepness_nm)}",
        "label": f"Steep Shortpass {int(cutoff_nm)} nm (edge {int(steepness_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def notch(center_nm: float, fwhm_nm: float = 10.0) -> dict:
    """Notch (band-reject) filter with Gaussian rejection profile."""
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    rejection = np.exp(-0.5 * ((WL_NM - center_nm) / sigma) ** 2)
    t = 1.0 - rejection
    r = rejection
    return {
        "name": f"notch_{int(center_nm)}_fwhm{int(fwhm_nm)}",
        "label": f"Notch {int(center_nm)} nm (FWHM {int(fwhm_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def dichroic_edge(cutoff_nm: float, steepness_nm: float = 5.0) -> dict:
    """Dichroic (interference) edge filter — extremely steep transition, all reflected."""
    t = 1.0 / (1.0 + np.exp(-(WL_NM - cutoff_nm) / (steepness_nm / 4.0)))
    r = 1.0 - t
    return {
        "name": f"dichroic_edge_{int(cutoff_nm)}_s{int(steepness_nm)}",
        "label": f"Dichroic Edge {int(cutoff_nm)} nm (edge {int(steepness_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def dichroic_bandpass(lo_nm: float, hi_nm: float, steepness_nm: float = 5.0) -> dict:
    """Dichroic bandpass filter — steep edges, reflected outside passband."""
    k = steepness_nm / 4.0
    t_lo = 1.0 / (1.0 + np.exp(-(WL_NM - lo_nm) / k))
    t_hi = 1.0 / (1.0 + np.exp((WL_NM - hi_nm) / k))
    t = t_lo * t_hi
    r = 1.0 - t
    return {
        "name": f"dichroic_bp_{int(lo_nm)}_{int(hi_nm)}_s{int(steepness_nm)}",
        "label": f"Dichroic Bandpass {int(lo_nm)}–{int(hi_nm)} nm (edge {int(steepness_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def neutral_density(od: float = 1.0) -> dict:
    """Neutral density filter — flat transmittance T = 10^(-OD), absorptive (R ≈ 0)."""
    t_val = 10.0 ** (-od)
    t = np.full(N_WL, t_val)
    r = np.zeros(N_WL)  # absorptive ND: energy absorbed, not reflected
    return {
        "name": f"nd_od{od:.1f}",
        "label": f"Neutral Density OD {od:.1f} (T={t_val:.1%})",
        "spectrum": _make_spectrum(r, t),
    }


def neutral_density_reflective(od: float = 1.0) -> dict:
    """Reflective neutral density filter — flat transmittance, remainder reflected."""
    t_val = 10.0 ** (-od)
    t = np.full(N_WL, t_val)
    r = np.full(N_WL, 1.0 - t_val)
    return {
        "name": f"nd_refl_od{od:.1f}",
        "label": f"Reflective ND OD {od:.1f} (T={t_val:.1%})",
        "spectrum": _make_spectrum(r, t),
    }


def multi_bandpass(bands: list[tuple[float, float]], steepness_nm: float = 5.0) -> dict:
    """Multi-bandpass filter — several pass-bands, blocking between them."""
    k = steepness_nm / 4.0
    t = np.zeros(N_WL)
    for lo, hi in bands:
        band = 1.0 / (1.0 + np.exp(-(WL_NM - lo) / k)) * 1.0 / (1.0 + np.exp((WL_NM - hi) / k))
        t = np.maximum(t, band)
    r = 1.0 - t
    band_str = "_".join(f"{int(lo)}-{int(hi)}" for lo, hi in bands)
    return {
        "name": f"multi_bp_{band_str}",
        "label": f"Multi-Bandpass {', '.join(f'{int(lo)}–{int(hi)}' for lo, hi in bands)} nm",
        "spectrum": _make_spectrum(r, t),
    }


def hot_mirror(cutoff_nm: float = 700.0, steepness_nm: float = 10.0) -> dict:
    """Hot mirror (IR cut-off) — transmits visible, reflects IR."""
    return {
        **steep_shortpass(cutoff_nm, steepness_nm),
        "name": f"hot_mirror_{int(cutoff_nm)}",
        "label": f"Hot Mirror (IR cut-off {int(cutoff_nm)} nm)",
    }


def cold_mirror(cutoff_nm: float = 700.0, steepness_nm: float = 10.0) -> dict:
    """Cold mirror — reflects visible, transmits IR."""
    return {
        **steep_longpass(cutoff_nm, steepness_nm),
        "name": f"cold_mirror_{int(cutoff_nm)}",
        "label": f"Cold Mirror (vis reflect, IR pass at {int(cutoff_nm)} nm)",
    }


def linear_variable(wl_start_nm: float = 500.0, wl_end_nm: float = 900.0,
                     fwhm_nm: float = 20.0) -> dict:
    """Linear variable filter — center wavelength sweeps across the grid.

    Simulated as a broadened peak whose center shifts linearly; each grid point
    sees a Gaussian centred at its 'local CWL' along the spatial axis.
    Since we have a single spectrum (no spatial dimension), we represent this as
    a broad plateau between wl_start and wl_end with Gaussian roll-off edges.
    """
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t_lo = 1.0 / (1.0 + np.exp(-(WL_NM - wl_start_nm) / sigma))
    t_hi = 1.0 / (1.0 + np.exp((WL_NM - wl_end_nm) / sigma))
    t = t_lo * t_hi
    r = 1.0 - t
    return {
        "name": f"lvf_{int(wl_start_nm)}_{int(wl_end_nm)}",
        "label": f"Linear Variable Filter {int(wl_start_nm)}–{int(wl_end_nm)} nm",
        "spectrum": _make_spectrum(r, t),
    }


# ── Registry ─────────────────────────────────────────────────────────────────

HANDCRAFTED_TARGETS: list[dict] = [
    # --- 1. Bandpass filters ---
    # Narrowband
    narrowband(532, fwhm_nm=10),   # laser line isolation (green)
    narrowband(633, fwhm_nm=10),   # HeNe laser line
    narrowband(850, fwhm_nm=20),   # NIR narrowband
    # Broadband
    broadband(550, fwhm_nm=100),   # green broadband
    broadband(750, fwhm_nm=150),   # NIR broadband
    bandpass(600, 900),            # original hard-edge bandpass

    # --- 2. Edge filters (longpass & shortpass) ---
    # Steep sigmoid edges
    steep_longpass(500, steepness_nm=10),
    steep_longpass(700, steepness_nm=10),
    steep_shortpass(600, steepness_nm=10),
    steep_shortpass(780, steepness_nm=10),
    # Original hard-edge versions
    longpass(600),
    shortpass(780),

    # --- 3. Notch filters (band-reject) ---
    notch(532, fwhm_nm=10),       # reject 532 nm laser line
    notch(633, fwhm_nm=15),       # reject HeNe laser line
    notch(785, fwhm_nm=10),       # reject 785 nm Raman excitation
    bandstop(520, 550),           # original hard-edge notch

    # --- 4. Dichroic filters ---
    dichroic_edge(650, steepness_nm=5),             # classic red/green dichroic
    dichroic_edge(500, steepness_nm=5),             # blue/green dichroic
    dichroic_bandpass(500, 600, steepness_nm=5),    # green passband dichroic

    # --- 5. Neutral density filters ---
    neutral_density(0.3),          # T ≈ 50 %
    neutral_density(0.5),          # T ≈ 32 %
    neutral_density(1.0),          # T ≈ 10 %
    neutral_density_reflective(0.5),
    neutral_density_reflective(1.0),

    # --- 6. Specialized & combined ---
    # Multi-bandpass
    multi_bandpass([(450, 500), (600, 650), (800, 850)]),   # RGB-like triple band
    multi_bandpass([(500, 540), (620, 660)]),                # dual fluorescence
    # Hot / cold mirrors
    hot_mirror(700),               # transmit visible, reflect IR
    cold_mirror(700),              # reflect visible, transmit IR
    # Linear variable filter
    linear_variable(500, 900),
]
