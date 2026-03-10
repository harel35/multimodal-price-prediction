"""Fusion modules for combining image and text embeddings."""

from .mlp import MLPFusion, create_mlp_fusion

__all__ = [
    "MLPFusion",
    "create_mlp_fusion",
]
