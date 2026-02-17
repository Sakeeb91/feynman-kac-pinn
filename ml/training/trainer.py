"""Training orchestration for Feynman-Kac PINN models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import torch

from ml.data.brownian import feynman_kac_estimate, get_device
from ml.data.domains import Domain
from ml.models.pinn import FeynmanKacPINN

from .losses import mc_mse_loss
from .schedulers import SchedulerName, WarmupConfig, build_scheduler


class BoundaryFn(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


class PotentialFn(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class FKProblem:
    domain: Domain
    boundary_condition: BoundaryFn
    potential: Optional[PotentialFn] = None


@dataclass
class TrainStepMetrics:
    loss: float
    lr: float
    grad_norm: float


@dataclass
class TrainerHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)


class FeynmanKacTrainer:
    """Training orchestrator for Feynman-Kac PINN."""

    def __init__(
        self,
        model: FeynmanKacPINN,
        problem: FKProblem,
        lr: float = 1e-3,
        device: Optional[str] = None,
        scheduler_name: SchedulerName = "none",
        warmup: WarmupConfig = WarmupConfig(),
    ):
        self.model = model
        self.problem = problem
        self.device = get_device() if device is None else device
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = build_scheduler(
            optimizer=self.optimizer,
            scheduler_name=scheduler_name,
            total_steps=1,
        )
        self.warmup = warmup
        self.history = TrainerHistory()
        self.global_step = 0


__all__ = [
    "FKProblem",
    "TrainStepMetrics",
    "TrainerHistory",
    "FeynmanKacTrainer",
]
