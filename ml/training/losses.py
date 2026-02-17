"""Loss functions for Feynman-Kac PINN training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mc_mse_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard mean-squared error against MC estimates."""
    return F.mse_loss(predictions, targets)


__all__ = ["mc_mse_loss"]
