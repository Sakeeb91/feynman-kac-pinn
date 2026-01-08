"""
Domain definitions for PDE problems.

Domains define the geometry over which PDEs are solved, including:
- Interior point checking
- Boundary projection for exit detection
"""

import torch
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class Domain(Protocol):
    """Protocol defining the interface for domain geometries."""

    @property
    def dim(self) -> int:
        """Spatial dimension of the domain."""
        ...

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside the domain.

        Args:
            x: Points to check, shape (batch_size, dim)

        Returns:
            Boolean mask, shape (batch_size,)
        """
        ...

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project points to the nearest boundary point.

        Used for exit detection when a Brownian path leaves the domain.

        Args:
            x: Points to project, shape (batch_size, dim)

        Returns:
            Projected points on boundary, shape (batch_size, dim)
        """
        ...

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample random points uniformly from the domain interior.

        Args:
            n: Number of points to sample
            device: Compute device

        Returns:
            Sampled points, shape (n, dim)
        """
        ...

    def sample_boundary(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample random points uniformly from the domain boundary.

        Args:
            n: Number of points to sample
            device: Compute device

        Returns:
            Sampled points, shape (n, dim)
        """
        ...


class Hypercube:
    """
    Axis-aligned hypercube domain [low, high]^dim.

    This is the most common domain for rectangular PDE problems
    like heat equations and Black-Scholes.
    """

    def __init__(self, low: float, high: float, dim: int):
        """
        Initialize hypercube domain.

        Args:
            low: Lower bound for all dimensions
            high: Upper bound for all dimensions
            dim: Spatial dimension
        """
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")
        if dim < 1:
            raise ValueError(f"dim ({dim}) must be positive")

        self._low = low
        self._high = high
        self._dim = dim

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
        """Check if points are strictly inside the hypercube."""
        return ((x > self._low) & (x < self._high)).all(dim=-1)

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp points to hypercube boundary."""
        return torch.clamp(x, self._low, self._high)

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Sample uniformly from hypercube interior."""
        return torch.rand(n, self._dim, device=device) * (self._high - self._low) + self._low

    def sample_boundary(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample uniformly from hypercube boundary (faces).

        Strategy: Pick a random face (2*dim faces total), then sample
        uniformly on that face.
        """
        points = torch.rand(n, self._dim, device=device) * (self._high - self._low) + self._low

        # Pick which dimension to fix (which face)
        face_dim = torch.randint(0, self._dim, (n,), device=device)
        # Pick which side (low or high)
        face_side = torch.randint(0, 2, (n,), device=device).float()

        # Set the face coordinate
        for i in range(n):
            d = face_dim[i].item()
            points[i, d] = self._low if face_side[i] == 0 else self._high

        return points

    def __repr__(self) -> str:
        return f"Hypercube(low={self._low}, high={self._high}, dim={self._dim})"


class Hypersphere:
    """
    Hypersphere domain {x : |x - center| < radius}.

    Useful for problems with radial symmetry or when testing
    against known analytical solutions for spherical domains.
    """

    def __init__(self, center: torch.Tensor | float, radius: float, dim: int):
        """
        Initialize hypersphere domain.

        Args:
            center: Center point (scalar for origin, or tensor of shape (dim,))
            radius: Radius of the sphere
            dim: Spatial dimension
        """
        if radius <= 0:
            raise ValueError(f"radius ({radius}) must be positive")
        if dim < 1:
            raise ValueError(f"dim ({dim}) must be positive")

        if isinstance(center, (int, float)):
            self._center = torch.full((dim,), float(center))
        else:
            self._center = center.clone()

        self._radius = radius
        self._dim = dim

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
        """Check if points are strictly inside the hypersphere."""
        center = self._center.to(x.device)
        dist_sq = ((x - center) ** 2).sum(dim=-1)
        return dist_sq < self._radius ** 2

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Project points to sphere surface (nearest point)."""
        center = self._center.to(x.device)
        direction = x - center
        dist = torch.norm(direction, dim=-1, keepdim=True)
        # Avoid division by zero for points at center
        dist = torch.clamp(dist, min=1e-10)
        return center + direction / dist * self._radius

    def sample_interior(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample uniformly from hypersphere interior.

        Uses rejection sampling from bounding hypercube.
        """
        center = self._center.to(device)
        samples = []
        while len(samples) < n:
            # Sample from bounding box
            candidates = (torch.rand(n * 2, self._dim, device=device) * 2 - 1) * self._radius + center
            inside = self.contains(candidates)
            samples.extend(candidates[inside].tolist())

        return torch.tensor(samples[:n], device=device)

    def sample_boundary(self, n: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample uniformly from hypersphere surface.

        Strategy: Sample from standard normal, normalize to unit sphere,
        then scale and translate.
        """
        center = self._center.to(device)
        # Sample from standard normal
        x = torch.randn(n, self._dim, device=device)
        # Normalize to unit sphere
        x = x / torch.norm(x, dim=-1, keepdim=True)
        # Scale and translate
        return x * self._radius + center

    def __repr__(self) -> str:
        return f"Hypersphere(center={self._center.tolist()}, radius={self._radius}, dim={self._dim})"


class Interval(Hypercube):
    """
    1D interval domain [low, high].

    Convenience class for 1D problems.
    """

    def __init__(self, low: float, high: float):
        super().__init__(low, high, dim=1)

    def __repr__(self) -> str:
        return f"Interval(low={self._low}, high={self._high})"
