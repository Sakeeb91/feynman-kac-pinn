"""Loss functions for Feynman-Kac PINN training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mc_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard mean-squared error against MC estimates."""
    return F.mse_loss(predictions, targets)


def weighted_mc_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE, useful when MC standard errors differ per sample."""
    if predictions.shape != targets.shape or predictions.shape != weights.shape:
        raise ValueError("predictions, targets, and weights must have matching shapes")
    weighted_sq = weights * (predictions - targets) ** 2
    return weighted_sq.mean()


def mc_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Robust alternative for noisy MC labels with occasional outliers."""
    return F.huber_loss(predictions, targets, delta=delta)


__all__ = ["mc_mse_loss", "weighted_mc_mse_loss", "mc_huber_loss"]
