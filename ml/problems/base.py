"""Abstract interfaces shared by PDE problem definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from ml.data.domains import Domain


class Problem(ABC):
    """Abstract base class for PDE problems solved via Feynman-Kac."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable problem name."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short textual summary for UIs/docs."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension."""

    @property
    @abstractmethod
    def domain(self) -> Domain:
        """Domain geometry where the PDE is solved."""

    @abstractmethod
    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Boundary/terminal condition g(x)."""

    @abstractmethod
    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """Potential/reaction term c(x)."""

    def analytical_solution(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return known analytical solution if available."""
        return None

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return initialization parameters for serialization."""

    def validate_points(self, x: torch.Tensor) -> torch.Tensor:
        """Validate tensor shape and dimension, return possibly reshaped points."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"expected x to have shape (batch, dim), got {tuple(x.shape)}")
        if x.shape[1] != self.dimension:
            raise ValueError(
                f"expected point dimension {self.dimension}, got {x.shape[1]}"
            )
        return x

    @property
    def metadata(self) -> dict[str, Any]:
        """Problem metadata for registry/API layers."""
        return {
            "name": self.name,
            "description": self.description,
            "dimension": self.dimension,
            "parameters": self.get_parameters(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize problem using registry-friendly schema."""
        return {
            "class_name": self.__class__.__name__,
            **self.metadata,
        }
