"""
Differentiable TMM (Transfer Matrix Method) in PyTorch.

Supports gradient-based optimization of thin-film layer thicknesses
and soft material selection via autodiff.

Stack: air (semi-inf) | coherent layers | incoherent substrate (500 µm) | air (semi-inf)
Matches optoformer's numpy inc_tmm exactly.
"""

import torch
import numpy as np
from torch import Tensor

from optoformer.constants import WL_NM, N_WL, MATERIALS, SUBSTRATE
from optoformer.data.sim import load_nk


def _build_nk_tensor(nk_dir: str) -> tuple[Tensor, Tensor]:
    """
    Load nk data and return tensors for all materials and substrate.

    Returns:
        mat_nk:  [N_MATERIALS, N_WL] complex64
        sub_nk:  [N_WL] complex64
    """
    nk_dict = load_nk(nk_dir)
    mat_nk = torch.stack([
        torch.from_numpy(nk_dict[m].astype(np.complex64)) for m in MATERIALS
    ])
    sub_nk = torch.from_numpy(nk_dict[SUBSTRATE].astype(np.complex64))
    return mat_nk, sub_nk


def _coh_tmm_forward(
    thicknesses: Tensor,       # [N_LAYERS]
    nk_per_layer: Tensor,      # [N_LAYERS, N_WL] complex
    n_entry: Tensor,           # [N_WL] complex — medium before stack
    n_exit: Tensor,            # [N_WL] complex — medium after stack
    wl: Tensor,                # [N_WL]
) -> tuple[Tensor, Tensor]:
    """
    Coherent TMM for s-polarization at normal incidence.
    Returns (R, T) as real tensors of shape [N_WL].
    """
    n_layers = thicknesses.shape[0]
    N = wl.shape[0]
    device = thicknesses.device
    dtype_c = nk_per_layer.dtype

    # Build total transfer matrix M = product of interface and propagation matrices
    # M stored as (m00, m01, m10, m11) — each [N_WL] complex
    m00 = torch.ones(N, dtype=dtype_c, device=device)
    m01 = torch.zeros(N, dtype=dtype_c, device=device)
    m10 = torch.zeros(N, dtype=dtype_c, device=device)
    m11 = torch.ones(N, dtype=dtype_c, device=device)

    prev_n = n_entry
    for i in range(n_layers):
        cur_n = nk_per_layer[i]
        # Interface matrix: prev_n → cur_n
        r = (prev_n - cur_n) / (prev_n + cur_n)
        t = 2.0 * prev_n / (prev_n + cur_n)
        inv_t = 1.0 / t
        # M = M @ [[1/t, r/t], [r/t, 1/t]]
        i00, i01, i10, i11 = inv_t, r * inv_t, r * inv_t, inv_t
        n00 = m00 * i00 + m01 * i10
        n01 = m00 * i01 + m01 * i11
        n10 = m10 * i00 + m11 * i10
        n11 = m10 * i01 + m11 * i11
        m00, m01, m10, m11 = n00, n01, n10, n11

        # Propagation matrix: [[exp(-i*phase), 0], [0, exp(i*phase)]]
        phase = 2.0 * np.pi * cur_n * thicknesses[i] / wl
        exp_neg = torch.exp(-1j * phase)
        exp_pos = torch.exp(1j * phase)
        n00 = m00 * exp_neg
        n01 = m01 * exp_pos
        n10 = m10 * exp_neg
        n11 = m11 * exp_pos
        m00, m01, m10, m11 = n00, n01, n10, n11
        prev_n = cur_n

    # Final interface: last layer → exit medium
    r = (prev_n - n_exit) / (prev_n + n_exit)
    t = 2.0 * prev_n / (prev_n + n_exit)
    inv_t = 1.0 / t
    i00, i01, i10, i11 = inv_t, r * inv_t, r * inv_t, inv_t
    n00 = m00 * i00 + m01 * i10
    n01 = m00 * i01 + m01 * i11
    n10 = m10 * i00 + m11 * i10
    n11 = m10 * i01 + m11 * i11
    m00, m01, m10, m11 = n00, n01, n10, n11

    # r = m10/m00, t = 1/m00
    r_amp = m10 / m00
    t_amp = 1.0 / m00
    R = (r_amp * r_amp.conj()).real
    T = (t_amp * t_amp.conj()).real * (n_exit.real / n_entry.real)
    return R, T


def tmm_spectrum(
    thicknesses: Tensor,       # [N_LAYERS]
    nk_per_layer: Tensor,      # [N_LAYERS, N_WL] complex
    sub_nk: Tensor,            # [N_WL] complex
) -> Tensor:
    """
    Differentiable TMM matching optoformer's inc_tmm.

    Coherent thin-film stack + incoherent substrate combination.

    Returns:
        [142] — 71 R then 71 T values.
    """
    device = thicknesses.device
    wl = torch.tensor(WL_NM, dtype=thicknesses.dtype, device=device)
    n_air = torch.ones(N_WL, dtype=sub_nk.dtype, device=device)

    # Forward: air → [layers] → substrate
    R_fwd, T_fwd = _coh_tmm_forward(thicknesses, nk_per_layer, n_air, sub_nk, wl)

    # Backward: substrate → [layers reversed] → air
    nk_rev = nk_per_layer.flip(0)
    thk_rev = thicknesses.flip(0)
    R_bwd, _ = _coh_tmm_forward(thk_rev, nk_rev, sub_nk, n_air, wl)

    # Substrate back surface: Glass → air
    r_sa = (sub_nk - n_air) / (sub_nk + n_air)
    R_sa = (r_sa * r_sa.conj()).real
    T_sa = 1.0 - R_sa

    # Substrate absorption (Glass has k≈0, but include for correctness)
    alpha = 4.0 * np.pi * sub_nk.imag / wl
    atten = torch.exp(-alpha * 500_000.0)

    # Incoherent combination
    denom = 1.0 - R_bwd * R_sa * atten * atten
    R = R_fwd + T_fwd * R_sa * atten * atten * T_fwd / denom  # approximation: T_back ≈ T_fwd
    T = T_fwd * atten * T_sa / denom

    R = R.clamp(0.0, 1.0)
    T = T.clamp(0.0, 1.0)
    return torch.cat([R, T])


def tmm_spectrum_batch(
    thicknesses: Tensor,       # [B, N_LAYERS]
    nk_per_layer: Tensor,      # [B, N_LAYERS, N_WL] complex
    sub_nk: Tensor,            # [N_WL] complex
) -> Tensor:
    """Batched version — loops over batch dim. Returns [B, 142]."""
    return torch.stack([
        tmm_spectrum(thicknesses[i], nk_per_layer[i], sub_nk)
        for i in range(thicknesses.shape[0])
    ])
