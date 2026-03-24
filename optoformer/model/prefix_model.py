"""
Prefix-conditioned model: the spectrum is projected to a single token and
prepended to the decoder sequence, replacing both BOS and cross-attention.

The decoder uses only self-attention (no cross-attention), making it
structurally an encoder stack with causal masking.  The spectrum prefix
token sits at position 0 and all material/thickness tokens attend to it
(and to each other, causally) through standard self-attention.

Two RoPE position variants are supported (select via `pos_mode`):
  - "raw"    : positions = thk_val[i]       (nm per layer)
  - "cumsum" : positions = cumsum(thk_vals)  (cumulative depth in nm)
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from optoformer.constants import N_SPECTRUM

from .common import (
    EncoderLayer,
    SpectrumProjection,
)


def _build_positions(thk_vals: Tensor, pos_mode: str) -> Tensor:
    """Convert thickness values to RoPE positions according to `pos_mode`."""
    if pos_mode == "cumsum":
        return thk_vals.float().cumsum(dim=-1)
    return thk_vals.float()  # "raw"


class MaterialEmbedding(nn.Module):
    """Material-only embedding — thickness is handled via RoPE, not fused here."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.mat_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.scale = math.sqrt(d_model)

    def forward(self, mat_ids: Tensor, thk_vals: Tensor) -> Tensor:
        return self.mat_embed(mat_ids) * self.scale


class InverseModel(nn.Module):
    """
    Prefix-conditioned autoregressive model: target spectrum → thin-film structure.

    Instead of cross-attention, the spectrum is projected to a d_model vector
    and prepended to the sequence as position 0.  The decoder is a stack of
    encoder layers (self-attention only) with causal masking.

    The spectrum prefix replaces BOS — during training the input sequence is:
        [SPEC, mat1, mat2, …, matN]      (thicknesses: [0, thk1, thk2, …, thkN])
    and the target (shifted right) is:
        [mat1, mat2, …, matN, EOS]       (thicknesses: [thk1, thk2, …, thkN, 0])

    At inference, decoding starts with just [SPEC] and tokens are appended
    autoregressively.
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
        pos_mode: str = "cumsum",
    ):
        super().__init__()
        assert pos_mode in ("raw", "cumsum"), f"Unknown pos_mode: {pos_mode!r}"
        self.vocab_size = vocab_size
        self.pos_mode = pos_mode
        self.d_model = d_model

        self.spectrum_proj = SpectrumProjection(d_model, n_spectrum)
        self.embedding = MaterialEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
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
        tgt_mat: Tensor,                  # [B, T] long  — material IDs (BOS, mat1, …)
        tgt_thk: Tensor,                  # [B, T] float — thicknesses  (0, thk1, …)
        tgt_mask: Tensor | None = None,   # [B, T, T] bool — causal+pad mask
    ) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        The spectrum is projected and prepended to the embedded material sequence.
        A causal self-attention mask (expanded by one column/row for the prefix)
        allows all tokens to attend to the spectrum and to earlier tokens.

        Returns:
            mat_logits: [B, T, vocab_size]  — logits at each decoder position
            thk_pred:   [B, T]              — thickness prediction at each position
        """
        B, T = tgt_mat.shape

        # Spectrum prefix: [B, 1, d_model]
        spec_token = self.spectrum_proj(spectrum)  # [B, 1, d_model]

        # Material token embeddings: [B, T, d_model]
        x = self.embedding(tgt_mat, tgt_thk)

        # Concatenate: [SPEC, tok1, tok2, …] → [B, 1+T, d_model]
        x = torch.cat([spec_token, x], dim=1)

        # Build causal mask for the extended sequence [B, 1+T, 1+T]
        if tgt_mask is not None:
            # tgt_mask is [B, T, T] — extend to [B, 1+T, 1+T]
            # The spectrum prefix column: every position can attend to it
            spec_col = torch.ones(B, T, 1, dtype=torch.bool, device=tgt_mat.device)
            extended_lower = torch.cat([spec_col, tgt_mask], dim=2)  # [B, T, 1+T]
            # The spectrum prefix row: it attends only to itself
            spec_row = torch.zeros(B, 1, 1 + T, dtype=torch.bool, device=tgt_mat.device)
            spec_row[:, :, 0] = True
            mask = torch.cat([spec_row, extended_lower], dim=1)  # [B, 1+T, 1+T]
        else:
            mask = None

        # RoPE positions: spectrum at 0, then material positions
        spec_pos = torch.zeros(B, 1, device=tgt_thk.device)
        mat_pos = _build_positions(tgt_thk, self.pos_mode)
        positions = torch.cat([spec_pos, mat_pos], dim=1)  # [B, 1+T]

        for layer in self.layers:
            x = layer(x, mask, positions)

        x = self.norm(x)

        # Output heads operate on material positions only (skip spectrum prefix)
        x_mat = x[:, 1:, :]  # [B, T, d_model]
        mat_logits = self.mat_head(x_mat)              # [B, T, vocab_size]
        thk_pred = self.thk_head(x_mat).squeeze(-1)   # [B, T]
        return mat_logits, thk_pred
