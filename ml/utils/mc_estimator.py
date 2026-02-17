"""High-level Monte Carlo APIs for Feynman-Kac estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from ml.data.boundary import BoundaryCondition
from ml.data.brownian import feynman_kac_estimate, get_device
from ml.data.domains import Domain


ProgressCallback = Callable[[int, int, int], None]


@dataclass
class SolveOutput:
    estimates: torch.Tensor
    std_errors: torch.Tensor


class FeynmanKacSolver:
    """High-level solver using Feynman-Kac Monte Carlo."""

    def __init__(
        self,
        domain: Domain,
        boundary_condition: BoundaryCondition,
        potential: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dt: float = 0.001,
        max_steps: int = 10000,
        device: Optional[str] = None,
        antithetic: bool = False,
        stratified: bool = False,
        batch_size: Optional[int] = None,
    ):
        self.domain = domain
        self.boundary_condition = boundary_condition
        self.potential = potential
        self.dt = dt
        self.max_steps = max_steps
        self.device = get_device() if device is None else device
        self.antithetic = antithetic
        self.stratified = stratified
        self.batch_size = batch_size

    def solve(
        self,
        x: torch.Tensor,
        n_samples: int = 1000,
        return_std: bool = True,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> SolveOutput | torch.Tensor:
        """Compute Monte Carlo estimate at query points."""
        estimates, std_errors = feynman_kac_estimate(
            x=x,
            boundary_fn=self.boundary_condition,
            domain=self.domain,
            potential_fn=self.potential,
            n_paths=n_samples,
            dt=self.dt,
            max_steps=self.max_steps,
            device=self.device,
            antithetic=self.antithetic,
            stratified=self.stratified,
            batch_size=self.batch_size,
            progress_callback=progress_callback,
        )
        if return_std:
            return SolveOutput(estimates=estimates, std_errors=std_errors)
        return estimates
