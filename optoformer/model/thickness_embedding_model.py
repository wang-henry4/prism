"""
Thickness-embedding model: thickness is projected as a learned linear embedding
added to the material embedding; RoPE uses standard sequential integer positions.

    embed(i) = (Embedding_mat(mat_id[i]) + Linear_thk(thk_val[i] / THK_MAX)) × √d_model
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from optoformer.constants import N_SPECTRUM, THK_MAX

from .common import (
    DecoderLayer,
    EncoderLayer,
    SpectrumHead,
    SpectrumProjection,
)


class ThicknessEmbedding(nn.Module):
    """
    Fuses material identity and thickness into one embedding vector.

        embed(i) = (Embedding_mat(mat_id[i]) + Linear_thk(thk_val[i] / THK_MAX)) × √d_model
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.mat_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.thk_proj  = nn.Linear(1, d_model, bias=False)
        self.scale     = math.sqrt(d_model)

    def forward(self, mat_ids: Tensor, thk_vals: Tensor) -> Tensor:
        """
        Args:
            mat_ids:  [B, S] long
            thk_vals: [B, S] float (nm)
        Returns:
            [B, S, d_model]
        """
        mat_emb = self.mat_embed(mat_ids)                          # [B, S, d_model]
        thk_emb = self.thk_proj(thk_vals.unsqueeze(-1) / THK_MAX)   # [B, S, d_model]
        return (mat_emb + thk_emb) * self.scale


class ForwardModel(nn.Module):
    """
    Encoder-only transformer: thin-film structure → optical spectrum.

    A CLS token is prepended; its final hidden state is mapped to the spectrum.
    RoPE uses sequential integer positions (0 = CLS, 1…S = layers).
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
        self.d_model = d_model

        self.embedding     = ThicknessEmbedding(vocab_size, d_model)
        self.cls_token     = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers        = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm          = nn.LayerNorm(d_model)
        self.spectrum_head = SpectrumHead(d_model, n_spectrum)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "cls_token" in name:
                nn.init.zeros_(p)
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        mat_ids: Tensor,                 # [B, S]
        thk_vals: Tensor,                # [B, S]
        src_mask: Tensor | None = None,  # [B, 1, S]  True=non-PAD
    ) -> Tensor:
        B, S = mat_ids.shape

        x   = self.embedding(mat_ids, thk_vals)  # [B, S, d_model]
        cls = self.cls_token.expand(B, -1, -1)   # [B, 1, d_model]
        x   = torch.cat([cls, x], dim=1)         # [B, 1+S, d_model]

        if src_mask is not None:
            cls_mask = torch.ones(B, 1, 1, dtype=torch.bool, device=mat_ids.device)
            mask = torch.cat([cls_mask, src_mask], dim=2)  # [B, 1, 1+S]
        else:
            mask = None

        positions = (
            torch.arange(S + 1, device=mat_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
            .float()
        )

        for layer in self.layers:
            x = layer(x, mask, positions)

        x = self.norm(x)
        return self.spectrum_head(x[:, 0, :])  # [B, n_spectrum]


class InverseModel(nn.Module):
    """
    Autoregressive decoder: target spectrum → thin-film structure.

    The spectrum is projected to a single memory token; the decoder cross-attends
    to it and produces per-position material logits and thickness predictions.
    RoPE uses sequential integer positions.
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
        self.embedding     = ThicknessEmbedding(vocab_size, d_model)
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
        tgt_thk: Tensor,                  # [B, T] float
        tgt_mask: Tensor | None = None,   # [B, T, T]  True=keep (causal+pad)
    ) -> tuple[Tensor, Tensor]:
        B, T = tgt_mat.shape

        memory = self.spectrum_proj(spectrum)       # [B, 1, d_model]
        x      = self.embedding(tgt_mat, tgt_thk)  # [B, T, d_model]

        positions = (
            torch.arange(T, device=tgt_mat.device)
            .unsqueeze(0)
            .expand(B, -1)
            .float()
        )

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, None, positions)

        x = self.norm(x)
        mat_logits = self.mat_head(x)               # [B, T, vocab_size]
        thk_pred   = self.thk_head(x).squeeze(-1)   # [B, T]
        return mat_logits, thk_pred
