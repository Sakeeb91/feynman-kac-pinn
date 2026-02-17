"""Boundary condition primitives and utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch

from .domains import Domain, Hypercube, Hypersphere, HyperEllipsoid


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


def estimate_outward_normals(domain: Domain, x: torch.Tensor) -> torch.Tensor:
    """Estimate outward unit normals for boundary points."""
    if isinstance(domain, Hypercube):
        center = torch.full((domain.dim,), (domain.low + domain.high) / 2, device=x.device, dtype=x.dtype)
        delta = x - center
        dominant = delta.abs().argmax(dim=-1)
        normals = torch.zeros_like(x)
        rows = torch.arange(x.shape[0], device=x.device)
        normals[rows, dominant] = torch.sign(delta[rows, dominant]).clamp_min(0.0) * 2 - 1
        return normals

    if isinstance(domain, Hypersphere):
        center = domain.center.to(x.device, dtype=x.dtype)
        normals = x - center
        return normals / torch.norm(normals, dim=-1, keepdim=True).clamp_min(1e-12)

    if isinstance(domain, HyperEllipsoid):
        center = domain.center.to(x.device, dtype=x.dtype)
        radii = domain.radii.to(x.device, dtype=x.dtype)
        grad = (x - center) / (radii**2)
        return grad / torch.norm(grad, dim=-1, keepdim=True).clamp_min(1e-12)

    # Fallback: finite-difference normal from projection map.
    eps = 1e-4
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
    grads = []
    for d in range(x.shape[1]):
        plus = domain.project_to_boundary(x + eps * eye[d])
        minus = domain.project_to_boundary(x - eps * eye[d])
        grads.append((plus - minus) / (2 * eps))
    grad_stack = torch.stack(grads, dim=-1).sum(dim=1)
    return grad_stack / torch.norm(grad_stack, dim=-1, keepdim=True).clamp_min(1e-12)


class NeumannBC(FluxBoundaryCondition):
    """Fixed normal derivative boundary condition ∂u/∂n|∂Ω = h(x)."""

    def __init__(self, flux_fn: TensorFn):
        self._flux_fn = flux_fn

    def __call__(self, x: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
        values = self._flux_fn(x)
        if values.dim() > 1 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        return values
