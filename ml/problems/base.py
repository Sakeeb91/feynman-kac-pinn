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
