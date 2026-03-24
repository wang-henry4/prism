"""
Thickness-RoPE model: thickness is not embedded; instead the cumulative depth
(nm) is used directly as the RoPE position argument, encoding geometry in the
attention rotation rather than in the token embedding.

    positions = cumsum(thk_vals)  — cumulative depth in nm
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from optoformer.constants import N_SPECTRUM

from .common import (
    DecoderLayer,
    SpectrumProjection,
)


class MaterialEmbedding(nn.Module):
    """
    Material-only embedding — thickness is handled via RoPE, not fused here.

        embed(i) = Embedding_mat(mat_id[i]) × √d_model
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.mat_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale     = math.sqrt(d_model)

    def forward(self, mat_ids: Tensor, thk_vals: Tensor) -> Tensor:
        """
        Args:
            mat_ids:  [B, S] long
            thk_vals: [B, S] float (nm) — accepted but unused; thickness goes to RoPE
        Returns:
            [B, S, d_model]
        """
        return self.mat_embed(mat_ids) * self.scale


class InverseModel(nn.Module):
    """
    Autoregressive decoder: target spectrum → thin-film structure.

    The spectrum is projected to a single memory token; the decoder cross-attends
    to it and produces per-position material logits and thickness predictions.
    RoPE positions are cumulative depth in nm.
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
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.spectrum_proj = SpectrumProjection(d_model, n_spectrum)
        self.embedding     = MaterialEmbedding(vocab_size, d_model)
        self.layers        = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm     = nn.LayerNorm(d_model)
        self.mat_head = nn.Linear(d_model, vocab_size)
        self.thk_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        spectrum: Tensor,                 # [B, 142]
        tgt_mat: Tensor,                  # [B, T] long
        tgt_thk: Tensor,                  # [B, T] float (nm)
        tgt_mask: Tensor | None = None,   # [B, T, T]  True=keep (causal+pad)
    ) -> tuple[Tensor, Tensor]:
        B, T = tgt_mat.shape

        memory    = self.spectrum_proj(spectrum)       # [B, 1, d_model]
        x         = self.embedding(tgt_mat, tgt_thk)  # [B, T, d_model]
        positions = tgt_thk.float().cumsum(dim=-1)     # [B, T]

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, None, positions)

        x = self.norm(x)
        mat_logits = self.mat_head(x)               # [B, T, vocab_size]
        thk_pred   = self.thk_head(x).squeeze(-1)   # [B, T]
        return mat_logits, thk_pred
