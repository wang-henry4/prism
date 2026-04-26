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
        "name": f"lvf_{int(wl_start_nm)}_{int(wl_end_nm)}_fwhm{int(fwhm_nm)}",
        "label": (f"Linear Variable Filter {int(wl_start_nm)}–{int(wl_end_nm)} nm "
                  f"(FWHM {int(fwhm_nm)} nm)"),
        "spectrum": _make_spectrum(r, t),
    }


def beam_splitter(t_ratio: float = 0.5) -> dict:
    """Beam splitter — flat partial-reflection / partial-transmission split."""
    t = np.full(N_WL, t_ratio)
    r = np.full(N_WL, 1.0 - t_ratio)
    pct_t = int(round(t_ratio * 100))
    return {
        "name": f"beam_splitter_{pct_t}t",
        "label": f"Beam Splitter {pct_t}/{100 - pct_t} (T/R)",
        "spectrum": _make_spectrum(r, t),
    }


def ar_coating(min_r: float = 0.005) -> dict:
    """Anti-reflection coating — near-zero reflectance across the band."""
    r = np.full(N_WL, min_r)
    t = 1.0 - r
    return {
        "name": f"ar_coating_r{int(round(min_r * 1000))}ppt",
        "label": f"AR Coating (R≈{min_r:.1%})",
        "spectrum": _make_spectrum(r, t),
    }


def broadband_mirror(lo_nm: float = 400.0, hi_nm: float = 1100.0,
                     steepness_nm: float = 10.0) -> dict:
    """Broadband high-reflectivity mirror — R≈1 across a band, T≈0."""
    k = steepness_nm / 4.0
    r_lo = 1.0 / (1.0 + np.exp(-(WL_NM - lo_nm) / k))
    r_hi = 1.0 / (1.0 + np.exp((WL_NM - hi_nm) / k))
    r = r_lo * r_hi
    t = 1.0 - r
    return {
        "name": f"hr_mirror_{int(lo_nm)}_{int(hi_nm)}",
        "label": f"Broadband HR Mirror {int(lo_nm)}–{int(hi_nm)} nm",
        "spectrum": _make_spectrum(r, t),
    }


_COLOR_PRESETS = {
    "red":   (620.0, 80.0),
    "green": (540.0, 80.0),
    "blue":  (460.0, 70.0),
}


def color_filter(color: str) -> dict:
    """Bayer-like RGB color filter with a Gaussian transmission profile."""
    center, fwhm = _COLOR_PRESETS[color]
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t = np.exp(-0.5 * ((WL_NM - center) / sigma) ** 2)
    r = 1.0 - t
    return {
        "name": f"color_{color}",
        "label": f"{color.capitalize()} Color Filter (~{int(center)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def comb_filter(spacing_nm: float = 100.0, fwhm_nm: float = 20.0,
                start_nm: float = 450.0) -> dict:
    """Comb filter — regularly spaced narrow pass-bands."""
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    t = np.zeros(N_WL)
    center = start_nm
    while center <= WL_NM[-1]:
        t = np.maximum(t, np.exp(-0.5 * ((WL_NM - center) / sigma) ** 2))
        center += spacing_nm
    r = 1.0 - t
    return {
        "name": f"comb_d{int(spacing_nm)}_fwhm{int(fwhm_nm)}",
        "label": f"Comb Filter (Δ={int(spacing_nm)} nm, FWHM {int(fwhm_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def fabry_perot(period_nm: float = 50.0, finesse: float = 4.0,
                phase_nm: float = 0.0) -> dict:
    """Fabry–Perot etalon — Airy function with periodic transmission peaks."""
    coef = (2.0 * finesse / np.pi) ** 2
    phase = np.pi * (WL_NM - phase_nm) / period_nm
    t = 1.0 / (1.0 + coef * np.sin(phase) ** 2)
    r = 1.0 - t
    return {
        "name": f"fabry_perot_p{int(period_nm)}_F{int(finesse)}",
        "label": f"Fabry–Perot (period {int(period_nm)} nm, finesse {finesse:.1f})",
        "spectrum": _make_spectrum(r, t),
    }


def triangular_bandpass(center_nm: float, half_width_nm: float = 50.0) -> dict:
    """Triangular bandpass — linear roll-off on either side of center."""
    t = np.maximum(0.0, 1.0 - np.abs(WL_NM - center_nm) / half_width_nm)
    r = 1.0 - t
    return {
        "name": f"tri_bp_{int(center_nm)}_hw{int(half_width_nm)}",
        "label": f"Triangular Bandpass {int(center_nm)} nm (±{int(half_width_nm)} nm)",
        "spectrum": _make_spectrum(r, t),
    }


def asymmetric_bandpass(center_nm: float, lo_edge_nm: float = 30.0,
                        hi_edge_nm: float = 5.0) -> dict:
    """Bandpass with different sigmoid roll-off slopes on each edge."""
    k_lo = lo_edge_nm / 4.0
    k_hi = hi_edge_nm / 4.0
    t_lo = 1.0 / (1.0 + np.exp(-(WL_NM - (center_nm - lo_edge_nm)) / k_lo))
    t_hi = 1.0 / (1.0 + np.exp((WL_NM - (center_nm + hi_edge_nm)) / k_hi))
    t = t_lo * t_hi
    r = 1.0 - t
    return {
        "name": f"asym_bp_{int(center_nm)}_l{int(lo_edge_nm)}_h{int(hi_edge_nm)}",
        "label": (f"Asymmetric Bandpass {int(center_nm)} nm "
                  f"(low edge {int(lo_edge_nm)}, high edge {int(hi_edge_nm)} nm)"),
        "spectrum": _make_spectrum(r, t),
    }


def staircase(levels: list[float], boundaries_nm: list[float]) -> dict:
    """Staircase transmittance — flat plateaus separated by sharp steps."""
    assert len(levels) == len(boundaries_nm) + 1
    t = np.zeros(N_WL)
    edges = [WL_NM[0] - 1.0] + list(boundaries_nm) + [WL_NM[-1] + 1.0]
    for i, lvl in enumerate(levels):
        mask = (WL_NM >= edges[i]) & (WL_NM < edges[i + 1])
        t[mask] = lvl
    r = 1.0 - t
    bnd_str = "_".join(str(int(b)) for b in boundaries_nm)
    lvl_str = "-".join(f"{lvl:.2f}" for lvl in levels)
    return {
        "name": f"staircase_{bnd_str}",
        "label": f"Staircase T={lvl_str}",
        "spectrum": _make_spectrum(r, t),
    }


def photopic_like() -> dict:
    """Photopic luminosity-like spectrum — eye sensitivity peak near 555 nm."""
    return {
        **broadband(555, fwhm_nm=110),
        "name": "photopic_like",
        "label": "Photopic Luminosity-like (peak 555 nm)",
    }


def qwot_stack_like(period_nm: float = 80.0, depth: float = 0.6,
                    offset: float = 0.5) -> dict:
    """QWOT stack-like sinusoidal reflectance fringes."""
    r = offset + depth * 0.5 * np.cos(2.0 * np.pi * WL_NM / period_nm)
    r = np.clip(r, 0.0, 1.0)
    t = 1.0 - r
    return {
        "name": f"qwot_p{int(period_nm)}_d{int(depth * 100)}",
        "label": f"QWOT Stack-like (period {int(period_nm)} nm, depth {depth:.2f})",
        "spectrum": _make_spectrum(r, t),
    }


# ── Registry ─────────────────────────────────────────────────────────────────

HANDCRAFTED_TARGETS: list[dict] = [
    # --- 1. Bandpass filters ---
    # Narrowband
    narrowband(450, fwhm_nm=5),    # ultra-narrow blue (atomic line)
    narrowband(532, fwhm_nm=10),   # laser line isolation (green)
    narrowband(633, fwhm_nm=10),   # HeNe laser line
    narrowband(656, fwhm_nm=3),    # H-alpha astronomy line
    narrowband(850, fwhm_nm=20),   # NIR narrowband
    narrowband(940, fwhm_nm=15),   # NIR LED isolation
    # Broadband
    broadband(450, fwhm_nm=80),    # blue broadband
    broadband(550, fwhm_nm=100),   # green broadband
    broadband(700, fwhm_nm=200),   # very broad red/NIR
    broadband(750, fwhm_nm=150),   # NIR broadband
    # Hard-edge bandpass
    bandpass(420, 480),            # blue hard-edge
    bandpass(500, 600),            # green hard-edge
    bandpass(600, 900),            # original hard-edge bandpass

    # --- 2. Edge filters (longpass & shortpass) ---
    # Steep sigmoid edges
    steep_longpass(450, steepness_nm=10),
    steep_longpass(500, steepness_nm=10),
    steep_longpass(700, steepness_nm=10),
    steep_longpass(850, steepness_nm=20),     # gentler edge
    steep_shortpass(550, steepness_nm=10),
    steep_shortpass(600, steepness_nm=10),
    steep_shortpass(780, steepness_nm=10),
    steep_shortpass(900, steepness_nm=20),    # gentler edge
    # Hard-edge versions
    longpass(500),
    longpass(600),
    longpass(800),
    shortpass(550),
    shortpass(780),
    shortpass(900),

    # --- 3. Notch filters (band-reject) ---
    notch(488, fwhm_nm=10),       # reject 488 nm Argon laser
    notch(532, fwhm_nm=10),       # reject 532 nm laser line
    notch(633, fwhm_nm=15),       # reject HeNe laser line
    notch(785, fwhm_nm=10),       # reject 785 nm Raman excitation
    notch(1064, fwhm_nm=20),      # reject Nd:YAG fundamental
    bandstop(520, 550),           # original hard-edge notch
    bandstop(700, 760),           # NIR hard-edge notch
    bandstop(450, 470),           # blue hard-edge notch

    # --- 4. Dichroic filters ---
    dichroic_edge(450, steepness_nm=5),             # UV/blue dichroic
    dichroic_edge(500, steepness_nm=5),             # blue/green dichroic
    dichroic_edge(650, steepness_nm=5),             # classic red/green dichroic
    dichroic_edge(800, steepness_nm=5),             # vis/NIR dichroic
    dichroic_bandpass(500, 600, steepness_nm=5),    # green passband dichroic
    dichroic_bandpass(620, 680, steepness_nm=3),    # tight red passband
    dichroic_bandpass(750, 900, steepness_nm=5),    # NIR passband dichroic

    # --- 5. Neutral density filters ---
    neutral_density(0.1),          # T ≈ 79 %
    neutral_density(0.3),          # T ≈ 50 %
    neutral_density(0.5),          # T ≈ 32 %
    neutral_density(1.0),          # T ≈ 10 %
    neutral_density(2.0),          # T ≈ 1 %
    neutral_density_reflective(0.3),
    neutral_density_reflective(0.5),
    neutral_density_reflective(1.0),
    neutral_density_reflective(2.0),

    # --- 6. Specialized & combined ---
    # Multi-bandpass
    multi_bandpass([(450, 500), (600, 650), (800, 850)]),   # RGB-like triple band
    multi_bandpass([(500, 540), (620, 660)]),                # dual fluorescence
    multi_bandpass([(430, 470), (520, 560), (610, 650)]),    # tight RGB Bayer-like
    multi_bandpass([(450, 480), (550, 580), (650, 680), (800, 850)]),  # quad band
    # Hot / cold mirrors
    hot_mirror(650),               # tighter visible window
    hot_mirror(700),               # transmit visible, reflect IR
    hot_mirror(750, steepness_nm=20),  # gentler edge
    cold_mirror(650),              # tighter
    cold_mirror(700),              # reflect visible, transmit IR
    cold_mirror(800),              # extended visible reflect
    # Linear variable filter — sweep wavelength range and bandwidth
    linear_variable(420, 700, fwhm_nm=20),    # visible-only LVF
    linear_variable(500, 900, fwhm_nm=20),    # original
    linear_variable(600, 1000, fwhm_nm=20),   # red-NIR LVF
    linear_variable(700, 1100, fwhm_nm=30),   # NIR-only LVF
    linear_variable(450, 850, fwhm_nm=10),    # narrow-bandwidth LVF
    linear_variable(450, 850, fwhm_nm=50),    # wide-bandwidth LVF

    # --- 7. Beam splitters & AR coatings ---
    beam_splitter(0.5),            # canonical 50/50 splitter
    beam_splitter(0.3),            # 30T / 70R (pellicle-like)
    beam_splitter(0.7),            # 70T / 30R
    beam_splitter(0.1),            # 10/90 sampling beam splitter
    beam_splitter(0.9),            # 90/10 sampling beam splitter
    ar_coating(0.001),             # premium AR (R≈0.1%)
    ar_coating(0.005),             # broadband AR (R≈0.5%)
    ar_coating(0.02),              # economy AR (R≈2%)

    # --- 8. Broadband mirrors ---
    broadband_mirror(400, 500),    # blue-only HR
    broadband_mirror(450, 700),    # visible HR mirror
    broadband_mirror(500, 900),    # vis-NIR HR
    broadband_mirror(700, 1100),   # NIR HR mirror

    # --- 9. RGB color filters (Bayer-like) ---
    color_filter("red"),
    color_filter("green"),
    color_filter("blue"),

    # --- 10. Periodic & interference-like spectra ---
    comb_filter(spacing_nm=50, fwhm_nm=10, start_nm=425),    # dense comb
    comb_filter(spacing_nm=100, fwhm_nm=20, start_nm=450),   # original
    comb_filter(spacing_nm=150, fwhm_nm=30, start_nm=475),   # sparse comb
    fabry_perot(period_nm=30, finesse=2.0),                  # broad FSR, low finesse
    fabry_perot(period_nm=50, finesse=4.0),
    fabry_perot(period_nm=80, finesse=8.0),
    fabry_perot(period_nm=120, finesse=15.0),                # sharp peaks
    qwot_stack_like(period_nm=60, depth=0.4),
    qwot_stack_like(period_nm=80, depth=0.6),
    qwot_stack_like(period_nm=120, depth=0.8),
    qwot_stack_like(period_nm=200, depth=0.9),               # slow modulation, deep

    # --- 11. Triangular & asymmetric profiles ---
    triangular_bandpass(500, half_width_nm=30),
    triangular_bandpass(650, half_width_nm=50),
    triangular_bandpass(800, half_width_nm=80),
    triangular_bandpass(950, half_width_nm=100),
    asymmetric_bandpass(700, lo_edge_nm=40, hi_edge_nm=5),   # Raman long-pass-ish
    asymmetric_bandpass(550, lo_edge_nm=10, hi_edge_nm=60),
    asymmetric_bandpass(850, lo_edge_nm=80, hi_edge_nm=10),

    # --- 12. Multi-level & vision-inspired ---
    staircase([0.0, 0.5, 1.0, 0.2], [550.0, 700.0, 900.0]),
    staircase([1.0, 0.25, 1.0], [600.0, 800.0]),
    staircase([0.2, 0.6, 1.0], [600.0, 850.0]),
    staircase([1.0, 0.7, 0.4, 0.1], [500.0, 700.0, 900.0]),  # descending
    photopic_like(),
]
