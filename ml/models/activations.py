"""Activation helpers for PINN models."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def swish(x: torch.Tensor) -> torch.Tensor:
    """Functional swish activation."""
    return x * torch.sigmoid(x)


def get_activation(name: str) -> nn.Module:
    """Return activation module by name."""
    key = name.lower()
    table: dict[str, Callable[[], nn.Module]] = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "identity": nn.Identity,
    }
    if key not in table:
        valid = ", ".join(sorted(table))
        raise ValueError(f"unsupported activation '{name}'. expected one of: {valid}")
    return table[key]()


def available_activations() -> tuple[str, ...]:
    """Return supported activation names for config validation/UI."""
    return ("gelu", "identity", "relu", "silu", "swish", "tanh")


__all__ = ["swish", "get_activation", "available_activations"]
