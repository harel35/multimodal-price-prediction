"""
Metrics and loss functions for the deep learning project.
"""
import torch


def smape_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    numerator = (preds - targets).abs()
    denominator = (preds.abs() + targets.abs()).clamp(min=eps) / 2.0
    return (numerator / denominator).mean()
