"""
Transformer models for thin-film optical design (inverse only).

Model variants (kept as separate modules):
  thickness_embedding_model  – "Thickness Embedding": thickness fused into token embedding; sequential RoPE
  thickness_rope_model       – "RoPE Thickness Encoding": thickness as RoPE positions (physical nm depth)
  prefix_model               – "Prefix RoPE Thickness Encoding": spectrum prefix + causal self-attention + RoPE depth
  prefix_material_thk_model  – "Augmented Regression RoPE Thickness Encoding": same as prefix, but thickness head
                                outputs per-material predictions via multi-layer MLP

Shared building blocks live in common.py.
"""

from .common import (
    apply_rope,
    DecoderLayer,
    FeedForward,
    MultiHeadAttention,
    ResidualConnection,
    SpectrumProjection,
)

from .prefix_material_thk_model import InverseModel
