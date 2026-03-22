"""
Transformer models for thin-film optical design.

Implementation is split across:
  common.py                  – shared building blocks
  thickness_embedding_model  – thickness fused into the token embedding; RoPE uses sequential positions
  thickness_rope_model       – thickness used directly as RoPE positions (physical nm depth)

Factory functions dispatch on config["arch"]:
  "A" → thickness_embedding_model
  "B" → thickness_rope_model
"""

from optoformer.constants import N_SPECTRUM

from .common import (
    apply_rope,
    EncoderLayer,
    DecoderLayer,
    FeedForward,
    MultiHeadAttention,
    ResidualConnection,
    SpectrumHead,
    SpectrumProjection,
)

import optoformer.model.thickness_embedding_model as thickness_embedding_model
import optoformer.model.thickness_rope_model as thickness_rope_model
import optoformer.model.prefix_model as prefix_model


# ── Factory functions ──────────────────────────────────────────────────────────

def make_forward_model(vocab_size: int, config: dict):
    arch = config.get("arch", "A")
    kwargs = dict(
        vocab_size = vocab_size,
        d_model    = config.get("d_model", 512),
        n_layers   = config.get("n_layers", 6),
        n_heads    = config.get("n_heads", 8),
        d_ff       = config.get("d_ff", 2048),
        dropout    = config.get("dropout", 0.1),
        n_spectrum = config.get("n_spectrum", N_SPECTRUM),
    )
    if arch == "A":
        return thickness_embedding_model.ForwardModel(**kwargs)
    if arch == "B":
        return thickness_rope_model.ForwardModel(**kwargs, pos_mode=config.get("pos_mode", "cumsum"))
    if arch == "C":
        return prefix_model.ForwardModel(**kwargs, pos_mode=config.get("pos_mode", "cumsum"))
    raise ValueError(f"Unknown arch: {arch!r}. Expected 'A', 'B', or 'C'.")


def make_inverse_model(vocab_size: int, config: dict):
    arch = config.get("arch", "A")
    kwargs = dict(
        vocab_size = vocab_size,
        d_model    = config.get("d_model", 512),
        n_layers   = config.get("n_layers", 6),
        n_heads    = config.get("n_heads", 8),
        d_ff       = config.get("d_ff", 2048),
        dropout    = config.get("dropout", 0.1),
        n_spectrum = config.get("n_spectrum", N_SPECTRUM),
    )
    if arch == "A":
        return thickness_embedding_model.InverseModel(**kwargs)
    if arch == "B":
        return thickness_rope_model.InverseModel(**kwargs, pos_mode=config.get("pos_mode", "cumsum"))
    if arch == "C":
        return prefix_model.InverseModel(**kwargs, pos_mode=config.get("pos_mode", "cumsum"))
    raise ValueError(f"Unknown arch: {arch!r}. Expected 'A', 'B', or 'C'.")
