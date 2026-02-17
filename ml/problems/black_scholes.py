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
        self._validate_params(params)
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

    @staticmethod
    def _validate_params(params: BlackScholesParams) -> None:
        if params.dim <= 0:
            raise ValueError("dim must be positive")
        if params.strike <= 0 or params.initial_price <= 0:
            raise ValueError("strike and initial_price must be positive")
        if params.maturity <= 0:
            raise ValueError("maturity must be positive")
        if params.volatility <= 0:
            raise ValueError("volatility must be positive")
        if not -1.0 < params.correlation < 1.0:
            raise ValueError("correlation must be in (-1, 1)")
        if params.option_type not in {"call", "put"}:
            raise ValueError("option_type must be either 'call' or 'put'")

    def _price_from_log(self, x: torch.Tensor) -> torch.Tensor:
        x = self.validate_points(x)
        s0 = torch.as_tensor(self._params.initial_price, device=x.device, dtype=x.dtype)
        return s0 * torch.exp(x)

    def _basket_payoff(self, basket_price: torch.Tensor) -> torch.Tensor:
        strike = torch.as_tensor(self._params.strike, device=basket_price.device, dtype=basket_price.dtype)
        if self._params.option_type == "call":
            return torch.clamp(basket_price - strike, min=0.0)
        return torch.clamp(strike - basket_price, min=0.0)

    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        prices = self._price_from_log(x)
        basket_price = prices.mean(dim=-1)
        return self._basket_payoff(basket_price)

    def potential(self, x: torch.Tensor) -> torch.Tensor:
        x = self.validate_points(x)
        return torch.full(
            (x.shape[0],),
            float(self._params.risk_free_rate),
            device=x.device,
            dtype=x.dtype,
        )

    def get_parameters(self) -> dict[str, float | int | str]:
        return self._params.__dict__.copy()
