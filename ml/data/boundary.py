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
