"""Boundary condition primitives and utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch


TensorFn = Callable[[torch.Tensor], torch.Tensor]


class BoundaryCondition(ABC):
    """Abstract base class for PDE boundary conditions."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate boundary operator at boundary points."""


class FluxBoundaryCondition(BoundaryCondition, ABC):
    """Boundary condition that may depend on outward unit normals."""

    @abstractmethod
    def __call__(self, x: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
        """Evaluate boundary flux at boundary points."""


class DirichletBC(BoundaryCondition):
    """Fixed value boundary condition u|∂Ω = g(x)."""

    def __init__(self, value_fn: TensorFn):
        self._value_fn = value_fn

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        values = self._value_fn(x)
        if values.dim() > 1 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        return values
