"""
Shared transformer building blocks used by both Architecture A and Architecture B.

Components:
  - apply_rope         – Rotary Position Embedding
  - MultiHeadAttention – RoPE-aware multi-head attention
  - FeedForward        – two-layer MLP with GELU
  - ResidualConnection – pre-norm residual wrapper
  - EncoderLayer       – self-attention + feed-forward
  - DecoderLayer       – self-attention + cross-attention + feed-forward
  - SpectrumHead       – CLS hidden state → spectrum floats
  - SpectrumProjection – spectrum floats → memory token for cross-attention
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from optoformer.constants import N_SPECTRUM


# ── Rotary Position Embedding ──────────────────────────────────────────────────

ROPE_BASE = 10000.0


def apply_rope(
    x: Tensor,
    positions: Tensor,
    scale_method: str = "none",
    scale_factor: float = 1.0,
) -> tuple[Tensor, float]:
    """
    Apply Rotary Position Embedding to x with optional context extension.

    Args:
        x:            [B, H, S, d_head]
        positions:    [B, S] float positions (cumulative depth in nm)
        scale_method: "none" | "pi" | "ntk" | "dynamic_ntk" | "yarn"
        scale_factor: Extension ratio (e.g. 2.0 for 2x OOD positions)

    Returns:
        rotated:   [B, H, S, d_head]  (same shape, rotated)
        attn_temp: scalar to multiply attention logits by (1.0 unless yarn)
    """
    d = x.shape[3]
    half = d // 2
    attn_temp = 1.0

    dim_idx = torch.arange(0, half, device=x.device, dtype=x.dtype) / half

    if scale_method == "pi":
        # Position Interpolation: compress positions into training range
        positions = positions / scale_factor
        theta = 1.0 / (ROPE_BASE ** dim_idx)
    elif scale_method == "ntk":
        # NTK-aware: scale the base frequency
        base = ROPE_BASE * scale_factor ** (d / (d - 2))
        theta = 1.0 / (base ** dim_idx)
    elif scale_method == "dynamic_ntk":
        # Dynamic NTK: compute scale from actual max position in batch
        max_pos = positions.max().item()
        # Training max cumulative depth: MAX_LAYERS * THK_MAX = 20 * 500 = 10000
        train_max = 10000.0
        if max_pos > train_max:
            dynamic_scale = max_pos / train_max
            base = ROPE_BASE * dynamic_scale ** (d / (d - 2))
        else:
            base = ROPE_BASE
        theta = 1.0 / (base ** dim_idx)
    elif scale_method == "yarn":
        # YaRN: NTK-aware base scaling + attention temperature correction
        base = ROPE_BASE * scale_factor ** (d / (d - 2))
        theta = 1.0 / (base ** dim_idx)
        attn_temp = 1.0 / math.sqrt(scale_factor)
    else:
        theta = 1.0 / (ROPE_BASE ** dim_idx)

    freqs = positions.unsqueeze(-1) * theta.unsqueeze(0).unsqueeze(0)  # [B, S, half]
    cos = freqs.cos().unsqueeze(1)  # [B, 1, S, half]
    sin = freqs.sin().unsqueeze(1)  # [B, 1, S, half]

    x1 = x[..., :half]  # [B, H, S, half]
    x2 = x[..., half:]  # [B, H, S, half]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated, attn_temp


# ── Building blocks ────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE applied to Q and K.
    Projection names: w_q, w_k, w_v, w_o.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,                           # [B, Sq, d_model]
        k: Tensor,                           # [B, Sk, d_model]
        v: Tensor,                           # [B, Sk, d_model]
        mask: Tensor | None = None,          # [B, Sq, Sk] or [B, 1, Sk]  True=keep
        positions: Tensor | None = None,     # [B, Sq] RoPE positions for Q (and K in self-attn)
        kv_positions: Tensor | None = None,  # [B, Sk] RoPE positions for K in cross-attn
        rope_scale_method: str = "none",
        rope_scale_factor: float = 1.0,
    ) -> Tensor:
        B, Sq = q.shape[0], q.shape[1]
        Sk = k.shape[1]

        Q = self.w_q(q).view(B, Sq, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Sq,d_head]
        K = self.w_k(k).view(B, Sk, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Sk,d_head]
        V = self.w_v(v).view(B, Sk, self.n_heads, self.d_head).transpose(1, 2)  # [B,H,Sk,d_head]

        attn_temp = 1.0
        if positions is not None:
            Q, attn_temp = apply_rope(Q, positions, rope_scale_method, rope_scale_factor)
            K, _ = apply_rope(K, kv_positions if kv_positions is not None else positions,
                              rope_scale_method, rope_scale_factor)

        scale = math.sqrt(self.d_head)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / scale * attn_temp  # [B, H, Sq, Sk]

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                              # [B, H, Sq, d_head]
        out = out.transpose(1, 2).contiguous().view(B, Sq, -1)  # [B, Sq, d_model]
        return self.w_o(out)


class FeedForward(nn.Module):
    """Linear → GELU → Dropout → Linear"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualConnection(nn.Module):
    """Pre-norm residual: x + dropout(sublayer(LayerNorm(x)))"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, sublayer) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


# ── Encoder / Decoder layers ───────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        positions: Tensor | None = None,
        rope_scale_method: str = "none",
        rope_scale_factor: float = 1.0,
    ) -> Tensor:
        x = self.res1(x, lambda z: self.self_attn(
            z, z, z, mask, positions,
            rope_scale_method=rope_scale_method, rope_scale_factor=rope_scale_factor,
        ))
        x = self.res2(x, self.ff)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff  = FeedForward(d_model, d_ff, dropout)
        self.res1 = ResidualConnection(d_model, dropout)
        self.res2 = ResidualConnection(d_model, dropout)
        self.res3 = ResidualConnection(d_model, dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        mem_mask: Tensor | None = None,
        positions: Tensor | None = None,
        rope_scale_method: str = "none",
        rope_scale_factor: float = 1.0,
    ) -> Tensor:
        x = self.res1(x, lambda z: self.self_attn(
            z, z, z, tgt_mask, positions,
            rope_scale_method=rope_scale_method, rope_scale_factor=rope_scale_factor,
        ))
        x = self.res2(x, lambda z: self.cross_attn(z, memory, memory, mem_mask))
        x = self.res3(x, self.ff)
        return x


# ── Spectrum I/O heads ─────────────────────────────────────────────────────────

class SpectrumProjection(nn.Module):
    """142-float spectrum → memory [B, 1, d_model] for cross-attention."""

    def __init__(self, d_model: int, n_spectrum: int = N_SPECTRUM):
        super().__init__()
        self.linear = nn.Linear(n_spectrum, d_model)

    def forward(self, spectrum: Tensor) -> Tensor:
        """spectrum: [B, n_spectrum] → [B, 1, d_model]"""
        return self.linear(spectrum).unsqueeze(1)
