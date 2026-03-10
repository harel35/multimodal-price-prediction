"""Loss and metric helpers used across training and evaluation."""

import torch


def smape_loss(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute SMAPE loss as ``mean(|p - y| / ((|p| + |y|)/2))``.

    Args:
        preds: Predicted prices, shape ``(batch,)`` or ``(batch, 1)``.
        targets: Ground-truth prices, same shape as ``preds``.
        eps: Numerical floor to avoid division by zero.

    Returns:
        Scalar tensor containing average SMAPE.
    """
    numerator = (preds - targets).abs()
    denominator = (preds.abs() + targets.abs()).clamp(min=eps) / 2.0
    return (numerator / denominator).mean()
