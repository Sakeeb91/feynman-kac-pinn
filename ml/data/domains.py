"""
Domain definitions for PDE problems.

Domains define the geometry over which PDEs are solved, including:
- Interior point checking
- Boundary projection for exit detection
- Uniform interior / boundary sampling
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


def _as_points(x: torch.Tensor, expected_dim: int) -> torch.Tensor:
    """Return `x` as shape (batch, dim) and validate dimensionality."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise ValueError(f"x must have shape (batch, dim), got {tuple(x.shape)}")
    if x.shape[1] != expected_dim:
        raise ValueError(f"expected dim={expected_dim}, got dim={x.shape[1]}")
    return x


def _check_sample_count(n: int) -> None:
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")


@runtime_checkable
class Domain(Protocol):
    """Protocol defining the interface for domain geometries."""

    @property
    def dim(self) -> int:
        ...

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        ...

    def sample_boundary(self, n: int, device: str = "cpu") -> torch.Tensor:
        ...

    def bounding_box(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Axis-aligned bounds used by generic sampling utilities."""
        ...


class Hypercube:
    """Axis-aligned hypercube domain [low, high]^dim."""

    def __init__(self, low: float, high: float, dim: int):
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        if dim < 1:
            raise ValueError(f"dim ({dim}) must be positive")

        self._low = float(low)
        self._high = float(high)
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_points(x, self._dim)
        return ((x > self._low) & (x < self._high)).all(dim=-1)

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_points(x, self._dim)
        return torch.clamp(x, self._low, self._high)

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        _check_sample_count(n)
        return torch.rand(n, self._dim, device=device) * (self._high - self._low) + self._low

    def sample_boundary(self, n: int, device: str = "cpu") -> torch.Tensor:
        _check_sample_count(n)
        points = torch.rand(n, self._dim, device=device) * (self._high - self._low) + self._low
        face_dim = torch.randint(0, self._dim, (n,), device=device)
        face_side = torch.randint(0, 2, (n,), device=device)
        rows = torch.arange(n, device=device)
        points[rows, face_dim] = torch.where(face_side == 0, self._low, self._high)
        return points

    def bounding_box(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        low = torch.full((self._dim,), self._low, device=device)
        high = torch.full((self._dim,), self._high, device=device)
        return low, high

    def __repr__(self) -> str:
        return f"Hypercube(low={self._low}, high={self._high}, dim={self._dim})"


class Hypersphere:
    """Hypersphere domain {x : ||x - center|| < radius}."""

    def __init__(self, center: torch.Tensor | float, radius: float, dim: int):
        if radius <= 0:
            raise ValueError(f"radius ({radius}) must be positive")
        if dim < 1:
            raise ValueError(f"dim ({dim}) must be positive")

        if isinstance(center, (int, float)):
            self._center = torch.full((dim,), float(center))
        else:
            center = center.clone().detach().float()
            if center.numel() != dim:
                raise ValueError(f"center must have dim={dim}, got {center.numel()}")
            self._center = center

        self._radius = float(radius)
        self._dim = int(dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def center(self) -> torch.Tensor:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_points(x, self._dim)
        center = self._center.to(x.device)
        dist_sq = ((x - center) ** 2).sum(dim=-1)
        return dist_sq < self._radius**2

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_points(x, self._dim)
        center = self._center.to(x.device)
        direction = x - center
        dist = torch.norm(direction, dim=-1, keepdim=True).clamp_min(1e-12)
        return center + direction / dist * self._radius

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        _check_sample_count(n)
        center = self._center.to(device)
        directions = torch.randn(n, self._dim, device=device)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True).clamp_min(1e-12)
        radii = torch.rand(n, 1, device=device).pow(1.0 / self._dim)
        return center + directions * radii * self._radius

    def sample_boundary(self, n: int, device: str = "cpu") -> torch.Tensor:
        _check_sample_count(n)
        center = self._center.to(device)
        x = torch.randn(n, self._dim, device=device)
        x = x / torch.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
        return x * self._radius + center

    def bounding_box(self, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        center = self._center.to(device)
        return center - self._radius, center + self._radius

    def __repr__(self) -> str:
        c = [round(v, 6) for v in self._center.tolist()]
        return f"Hypersphere(center={c}, radius={self._radius}, dim={self._dim})"


class Interval(Hypercube):
    """1D interval domain [low, high]."""

    def __init__(self, low: float, high: float):
        super().__init__(low, high, dim=1)

    def __repr__(self) -> str:
        return f"Interval(low={self._low}, high={self._high})"
