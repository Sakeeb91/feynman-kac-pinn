"""Main PINN architecture for Feynman-Kac approximation."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .activations import get_activation
from .initialization import InitMode, initialize_module


class FeynmanKacPINN(nn.Module):
    """Configurable MLP that approximates solution values u(x)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64, 64, 64),
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        init_mode: InitMode = "xavier",
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must not be empty")
        if any(width <= 0 for width in hidden_dims):
            raise ValueError("all hidden layer widths must be positive")

        self.input_dim = input_dim
        self.hidden_dims = tuple(int(width) for width in hidden_dims)
        self.activation_name = activation
        self.output_activation_name = output_activation

        blocks: list[nn.Module] = []
        prev = input_dim
        for width in self.hidden_dims:
            blocks.append(nn.Linear(prev, width))
            blocks.append(get_activation(activation))
            prev = width
        self.hidden = nn.Sequential(*blocks)
        self.output = nn.Linear(prev, 1)
        self.output_activation = get_activation(output_activation) if output_activation else None

        initialize_module(self, mode=init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected x shape (batch, {self.input_dim}), got {tuple(x.shape)}")
        y = self.hidden(x)
        y = self.output(y)
        if self.output_activation is not None:
            y = self.output_activation(y)
        return y


__all__ = ["FeynmanKacPINN"]
