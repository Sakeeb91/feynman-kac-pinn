"""Black-Scholes problem definitions for basket options."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ml.data.domains import Hypercube

from .base import Problem


@dataclass(frozen=True)
class BlackScholesParams:
    dim: int = 10
    strike: float = 100.0
    risk_free_rate: float = 0.05
    volatility: float = 0.2
    correlation: float = 0.3
    maturity: float = 1.0
    initial_price: float = 100.0
    option_type: str = "call"
    log_price_bound: float = 2.0


class BlackScholesND(Problem):
    """N-dimensional Black-Scholes basket option problem."""

    def __init__(self, **kwargs: float | int | str):
        params = BlackScholesParams(**kwargs)
        self._params = params
        self._domain = Hypercube(
            low=-params.log_price_bound,
            high=params.log_price_bound,
            dim=params.dim,
        )

    @property
    def name(self) -> str:
        return f"{self._params.dim}D Black-Scholes"

    @property
    def description(self) -> str:
        return "Basket option pricing under correlated geometric Brownian motion."

    @property
    def dimension(self) -> int:
        return self._params.dim

    @property
    def domain(self) -> Hypercube:
        return self._domain

    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_parameters(self) -> dict[str, float | int | str]:
        return self._params.__dict__.copy()
