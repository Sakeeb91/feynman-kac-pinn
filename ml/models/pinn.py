"""Main PINN architecture for Feynman-Kac approximation."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .activations import get_activation
from .initialization import InitMode, initialize_module


class _HiddenBlock(nn.Module):
    """Linear + activation block with optional residual connection."""

    def __init__(self, in_dim: int, out_dim: int, activation: str, use_residual: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = get_activation(activation)
        self.use_residual = use_residual and in_dim == out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        if self.use_residual:
            y = y + x
        return y


class FeynmanKacPINN(nn.Module):
    """Configurable MLP that approximates solution values u(x)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (64, 64, 64, 64),
        activation: str = "tanh",
        output_activation: Optional[str] = None,
        use_residual: bool = False,
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
        self.use_residual = use_residual

        blocks: list[nn.Module] = []
        prev = input_dim
        for width in self.hidden_dims:
            blocks.append(_HiddenBlock(prev, width, activation, use_residual=use_residual))
            prev = width
        self.hidden = nn.ModuleList(blocks)
        self.output = nn.Linear(prev, 1)
        self.output_activation = get_activation(output_activation) if output_activation else None

        initialize_module(self, mode=init_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected x shape (batch, {self.input_dim}), got {tuple(x.shape)}")
        y = x
        for block in self.hidden:
            y = block(y)
        y = self.output(y)
        if self.output_activation is not None:
            y = self.output_activation(y)
        return y

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return model parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def architecture_dict(self) -> dict[str, object]:
        """Serializable model config used by checkpoint metadata."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation_name,
            "output_activation": self.output_activation_name,
            "use_residual": self.use_residual,
        }


__all__ = ["FeynmanKacPINN"]
