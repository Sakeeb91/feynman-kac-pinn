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

    def correlation_matrix(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return constant-correlation matrix used by the GBM basket model."""
        rho = float(self._params.correlation)
        mat = torch.full((self.dimension, self.dimension), rho, device=device, dtype=dtype)
        idx = torch.arange(self.dimension, device=device)
        mat[idx, idx] = 1.0
        return mat

    def cholesky_correlation(
        self,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return Cholesky factor for correlated normal sampling."""
        corr = self.correlation_matrix(device=device, dtype=dtype)
        return torch.linalg.cholesky(corr)

    @staticmethod
    def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1.0 + torch.erf(x / (2.0**0.5)))

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

    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor | None:
        """Closed-form Black-Scholes formula for the 1D case only."""
        if self.dimension != 1:
            return None
        x = self.validate_points(x)
        s = self._price_from_log(x).squeeze(-1)
        k = torch.as_tensor(self._params.strike, dtype=s.dtype, device=s.device)
        r = torch.as_tensor(self._params.risk_free_rate, dtype=s.dtype, device=s.device)
        sigma = torch.as_tensor(self._params.volatility, dtype=s.dtype, device=s.device)
        t = torch.as_tensor(self._params.maturity, dtype=s.dtype, device=s.device)
        sqrt_t = torch.sqrt(t)
        d1 = (torch.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        discount = torch.exp(-r * t)
        if self._params.option_type == "call":
            return s * self._normal_cdf(d1) - k * discount * self._normal_cdf(d2)
        return k * discount * self._normal_cdf(-d2) - s * self._normal_cdf(-d1)

    def get_parameters(self) -> dict[str, float | int | str]:
        return self._params.__dict__.copy()
