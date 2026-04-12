"""
Design-space and simulation constants shared across the project.

All modules should import from here rather than defining these values locally.
"""

import numpy as np

# ── Wavelength grid ───────────────────────────────────────────────────────────
WL_START = 400       # nm
WL_END = 1100        # nm (inclusive)
WL_STEP = 10         # nm
WL_NM = np.arange(WL_START, WL_END + WL_STEP, WL_STEP, dtype=float)  # 71 points
WL_UM = WL_NM / 1000.0                                                # µm
N_WL = len(WL_NM)                                                     # 71
N_SPECTRUM = 2 * N_WL                                                 # 142 (R + T)

# ── Thickness ─────────────────────────────────────────────────────────────────
THK_MIN = 10          # nm
THK_MAX = 10*50        # nm
THK_STEP = 10         # nm

# ── Layer count ───────────────────────────────────────────────────────────────
MAX_LAYERS = 20
MIN_LAYERS = 1

# ── Materials ─────────────────────────────────────────────────────────────────
MATERIALS = [
    "Al", "Al2O3", "AlN", "Ge", "HfO2", "ITO", "MgF2", "MgO",
    "Si", "Si3N4", "SiO2", "Ta2O5", "TiN", "TiO2", "ZnO", "ZnS", "ZnSe",
]
SUBSTRATE = "Glass_Substrate"
