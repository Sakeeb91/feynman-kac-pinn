"""Schrodinger-type PDE problems for quantum ground states."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ml.data.domains import Hypersphere

from .base import Problem


@dataclass(frozen=True)
class HarmonicOscillatorParams:
    dim: int = 10
    domain_radius: float = 5.0
    mass: float = 1.0
    omega: float = 1.0
    hbar: float = 1.0


class HarmonicOscillatorND(Problem):
    """N-dimensional harmonic oscillator in configurable physical units."""

    def __init__(self, **kwargs: float | int):
        params = HarmonicOscillatorParams(**kwargs)
        self._validate_params(params)
        self._params = params
        self._domain = Hypersphere(center=0.0, radius=params.domain_radius, dim=params.dim)

    @staticmethod
    def _validate_params(params: HarmonicOscillatorParams) -> None:
        if params.dim <= 0:
            raise ValueError("dim must be positive")
        if params.domain_radius <= 0:
            raise ValueError("domain_radius must be positive")
        if params.mass <= 0 or params.omega <= 0 or params.hbar <= 0:
            raise ValueError("mass, omega, and hbar must be positive")

    @property
    def name(self) -> str:
        return f"{self._params.dim}D Harmonic Oscillator"

    @property
    def description(self) -> str:
        return "Ground-state harmonic oscillator with Gaussian eigenfunction."

    @property
    def dimension(self) -> int:
        return self._params.dim

    @property
    def domain(self) -> Hypersphere:
        return self._domain

    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        x = self.validate_points(x)
        return self.analytical_solution(x)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        x = self.validate_points(x)
        mass = torch.as_tensor(self._params.mass, device=x.device, dtype=x.dtype)
        omega = torch.as_tensor(self._params.omega, device=x.device, dtype=x.dtype)
        radius_sq = (x**2).sum(dim=-1)
        return 0.5 * mass * omega**2 * radius_sq

    @property
    def ground_state_energy(self) -> float:
        return 0.5 * self.dimension * self._params.hbar * self._params.omega

    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        x = self.validate_points(x)
        alpha = (self._params.mass * self._params.omega) / self._params.hbar
        norm = (alpha / torch.pi) ** (self.dimension / 4)
        radius_sq = (x**2).sum(dim=-1)
        return norm * torch.exp(-0.5 * alpha * radius_sq)

    def get_parameters(self) -> dict[str, float | int]:
        return self._params.__dict__.copy()
