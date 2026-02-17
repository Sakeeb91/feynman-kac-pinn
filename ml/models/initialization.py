"""Weight initialization helpers for PINN architectures."""

from __future__ import annotations

from typing import Literal

from torch import nn


InitMode = Literal["xavier", "he"]


def init_linear_layer(layer: nn.Linear, mode: InitMode = "xavier") -> None:
    """Initialize a linear layer with the requested scheme."""
    if mode == "xavier":
        nn.init.xavier_uniform_(layer.weight)
    elif mode == "he":
        nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    else:
        raise ValueError(f"unsupported init mode '{mode}'")
    nn.init.zeros_(layer.bias)


def initialize_module(module: nn.Module, mode: InitMode = "xavier") -> None:
    """Apply initialization recursively to all linear layers in a module."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            init_linear_layer(layer, mode=mode)


__all__ = ["InitMode", "init_linear_layer", "initialize_module"]
