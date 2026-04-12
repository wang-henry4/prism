"""
Transformer model for inverse thin-film optical design.

Architecture: spectrum prefix + causal self-attention + cumulative-depth RoPE +
per-material thickness MLP head.  Shared building blocks live in common.py.
"""

from .common import (
    apply_rope,
    EncoderLayer,
    FeedForward,
    MultiHeadAttention,
    ResidualConnection,
    SpectrumProjection,
)

from .prefix_material_thk_model import InverseModel
