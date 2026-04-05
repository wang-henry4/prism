"""
Prefix-conditioned model variant with per-material thickness predictions.

Same architecture as prefix_model (spectrum prefix, causal self-attention,
RoPE with cumulative depth), but the thickness head outputs a prediction
for every material in the vocab.  This allows beam search to jointly score
(material, thickness) pairs without committing to a material first.

The thickness head is a multi-layer MLP for added non-linearity.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from optoformer.constants import N_SPECTRUM, THK_MIN

from .common import (
    EncoderLayer,
    SpectrumProjection,
)


class MaterialEmbedding(nn.Module):
    """Material-only embedding — thickness is handled via RoPE, not fused here."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.mat_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale = math.sqrt(d_model)

    def forward(self, mat_ids: Tensor, thk_vals: Tensor) -> Tensor:
        return self.mat_embed(mat_ids) * self.scale


class ThicknessMLPHead(nn.Module):
    """
    Multi-layer MLP that predicts a thickness value per material.

    Input:  [B, T, d_model]
    Output: [B, T, vocab_size]

    Each output dimension corresponds to the predicted thickness for that
    material ID, enabling joint (material, thickness) beam search.

    When log_space=True, the raw output is passed through softplus to ensure
    positivity (representing log(thk / THK_MIN)).  The caller converts to nm
    via ``THK_MIN * exp(output)``.  When log_space=False (default), the raw
    linear output is returned directly (nm).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_hidden_layers: int = 2,
        d_hidden: int | None = None,
        dropout: float = 0.1,
        log_space: bool = True,
    ):
        super().__init__()
        self.log_space = log_space
        d_hidden = d_hidden or d_model
        layers: list[nn.Module] = []

        # Input layer
        layers.extend([nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout)])

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout)])

        # Output projection
        layers.append(nn.Linear(d_hidden, vocab_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        if self.log_space:
            # softplus ensures output > 0 (represents log(thk / THK_MIN))
            out = F.softplus(out)
        return out


class InverseModel(nn.Module):
    """
    Prefix-conditioned autoregressive model with per-material thickness head.

    Identical to prefix_model.InverseModel except:
      - thk_head outputs [B, T, vocab_size] instead of [B, T, 1]
      - thk_head is a multi-layer MLP (ThicknessMLPHead)

    The i-th output of thk_head at position t is the predicted thickness
    if material i is chosen at that position.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        n_spectrum: int = N_SPECTRUM,
        thk_head_hidden_layers: int = 2,
        log_space_thk: bool = True,
        rope_scale_method: str = "none",
        rope_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.log_space_thk = log_space_thk
        self.rope_scale_method = rope_scale_method
        self.rope_scale_factor = rope_scale_factor

        self.spectrum_proj = SpectrumProjection(d_model, n_spectrum)
        self.embedding = MaterialEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.mat_head = nn.Linear(d_model, vocab_size)
        self.thk_head = ThicknessMLPHead(
            d_model=d_model,
            vocab_size=vocab_size,
            n_hidden_layers=thk_head_hidden_layers,
            d_hidden=d_model,
            dropout=dropout,
            log_space=log_space_thk,
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        spectrum: Tensor,                 # [B, 142]
        tgt_mat: Tensor,                  # [B, T] long  — material IDs (BOS, mat1, …)
        tgt_thk: Tensor,                  # [B, T] float — thicknesses  (0, thk1, …)
        tgt_mask: Tensor | None = None,   # [B, T, T] bool — causal+pad mask
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Returns:
            mat_logits: [B, T, vocab_size]  — logits at each decoder position
            thk_pred:   [B, T, vocab_size]  — per-material thickness prediction.
                        When log_space_thk=True this is log(thk/THK_MIN) (use
                        thk_to_nm() to convert).  Otherwise raw nm.
        """
        B, T = tgt_mat.shape

        # Spectrum prefix: [B, 1, d_model]
        spec_token = self.spectrum_proj(spectrum)

        # Material token embeddings: [B, T, d_model]
        x = self.embedding(tgt_mat, tgt_thk)

        # Concatenate: [SPEC, tok1, tok2, …] → [B, 1+T, d_model]
        x = torch.cat([spec_token, x], dim=1)

        # Build causal mask for the extended sequence [B, 1+T, 1+T]
        if tgt_mask is not None:
            spec_col = torch.ones(B, T, 1, dtype=torch.bool, device=tgt_mat.device)
            extended_lower = torch.cat([spec_col, tgt_mask], dim=2)
            spec_row = torch.zeros(B, 1, 1 + T, dtype=torch.bool, device=tgt_mat.device)
            spec_row[:, :, 0] = True
            mask = torch.cat([spec_row, extended_lower], dim=1)
        else:
            mask = None

        # RoPE positions: spectrum at 0, then cumulative depth
        spec_pos = torch.zeros(B, 1, device=tgt_thk.device)
        mat_pos = tgt_thk.float().cumsum(dim=-1)
        positions = torch.cat([spec_pos, mat_pos], dim=1)

        for layer in self.layers:
            x = layer(x, mask, positions, self.rope_scale_method, self.rope_scale_factor)

        x = self.norm(x)

        # Output heads operate on material positions only (skip spectrum prefix)
        x_mat = x[:, 1:, :]                          # [B, T, d_model]
        mat_logits = self.mat_head(x_mat)             # [B, T, vocab_size]
        thk_pred = self.thk_head(x_mat)               # [B, T, vocab_size]
        return mat_logits, thk_pred

    def thk_to_nm(self, thk_pred: Tensor) -> Tensor:
        """Convert thickness head output to nm.

        No-op when log_space_thk=False.  When True, computes
        ``THK_MIN * exp(clamp(thk_pred, max=8))`` (~29 800 nm safety cap).
        """
        if not self.log_space_thk:
            return thk_pred
        return THK_MIN * torch.exp(thk_pred.clamp(max=8.0))

    @staticmethod
    def nm_to_log(thk_nm: Tensor) -> Tensor:
        """Convert nm thickness to log-space target: log(thk / THK_MIN)."""
        return torch.log(thk_nm.clamp(min=THK_MIN) / THK_MIN)
