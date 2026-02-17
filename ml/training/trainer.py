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

    def _sample_training_batch(self, batch_size: int, n_mc_paths: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.problem.domain.sample_interior(batch_size, device=self.device)
        targets, _ = feynman_kac_estimate(
            x=x,
            boundary_fn=self.problem.boundary_condition,
            domain=self.problem.domain,
            potential_fn=self.problem.potential,
            n_paths=n_mc_paths,
            device=self.device,
        )
        return x, targets

    def _compute_grad_norm(self) -> float:
        norm = torch.zeros((), device=self.device)
        for parameter in self.model.parameters():
            if parameter.grad is None:
                continue
            norm = norm + parameter.grad.detach().pow(2).sum()
        return torch.sqrt(norm).item()

    def train_step(self, batch_size: int, n_mc_paths: int) -> TrainStepMetrics:
        """Run one optimization step against fresh MC supervision."""
        self.model.train()
        x, targets = self._sample_training_batch(batch_size=batch_size, n_mc_paths=n_mc_paths)
        predictions = self.model(x).squeeze(-1)
        loss = mc_mse_loss(predictions, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.global_step += 1

        grad_norm = self._compute_grad_norm()
        current_lr = self.optimizer.param_groups[0]["lr"]
        metrics = TrainStepMetrics(loss=loss.item(), lr=current_lr, grad_norm=grad_norm)
        self.history.train_loss.append(metrics.loss)
        self.history.lr.append(metrics.lr)
        self.history.grad_norm.append(metrics.grad_norm)
        return metrics


__all__ = [
    "FKProblem",
    "TrainStepMetrics",
    "TrainerHistory",
    "FeynmanKacTrainer",
]
