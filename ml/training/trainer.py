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
from .schedulers import (
    SchedulerName,
    WarmupConfig,
    build_linear_warmup_scheduler,
    build_scheduler,
)


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


@dataclass
class EarlyStoppingState:
    patience: int = 25
    min_delta: float = 0.0
    best_loss: float = float("inf")
    bad_epochs: int = 0


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
        max_grad_norm: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_min_delta: float = 0.0,
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
        self.warmup_scheduler = (
            build_linear_warmup_scheduler(
                optimizer=self.optimizer,
                warmup_steps=warmup.steps,
                start_factor=warmup.start_factor,
            )
            if warmup.steps > 0
            else None
        )
        self.max_grad_norm = max_grad_norm
        self.history = TrainerHistory()
        self.global_step = 0
        self.early_stopping = (
            EarlyStoppingState(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
            if early_stopping_patience is not None
            else None
        )

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
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.warmup_scheduler is not None and self.global_step < self.warmup.steps:
            self.warmup_scheduler.step()
        elif self.scheduler is not None and not isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step()
        self.global_step += 1

        grad_norm = self._compute_grad_norm()
        current_lr = self.optimizer.param_groups[0]["lr"]
        metrics = TrainStepMetrics(loss=loss.item(), lr=current_lr, grad_norm=grad_norm)
        self.history.train_loss.append(metrics.loss)
        self.history.lr.append(metrics.lr)
        self.history.grad_norm.append(metrics.grad_norm)
        return metrics

    @torch.no_grad()
    def eval_step(self, batch_size: int, n_mc_paths: int) -> float:
        """Evaluate current model against fresh MC labels."""
        self.model.eval()
        x, targets = self._sample_training_batch(batch_size=batch_size, n_mc_paths=n_mc_paths)
        predictions = self.model(x).squeeze(-1)
        loss = mc_mse_loss(predictions, targets).item()
        return loss

    def validate(self, batch_size: int, n_mc_paths: int) -> float:
        val_loss = self.eval_step(batch_size=batch_size, n_mc_paths=n_mc_paths)
        self.history.val_loss.append(val_loss)
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        if self.early_stopping is not None:
            if val_loss < self.early_stopping.best_loss - self.early_stopping.min_delta:
                self.early_stopping.best_loss = val_loss
                self.early_stopping.bad_epochs = 0
            else:
                self.early_stopping.bad_epochs += 1
        return val_loss

    def should_stop_early(self) -> bool:
        if self.early_stopping is None:
            return False
        return self.early_stopping.bad_epochs >= self.early_stopping.patience

    def fit(
        self,
        steps: int,
        batch_size: int,
        n_mc_paths: int,
        val_interval: int = 0,
        val_batch_size: Optional[int] = None,
        val_mc_paths: Optional[int] = None,
        step_callback: Optional[Callable[[int, TrainStepMetrics], None]] = None,
    ) -> TrainerHistory:
        """Run repeated train steps with optional periodic validation."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        if batch_size <= 0 or n_mc_paths <= 0:
            raise ValueError("batch_size and n_mc_paths must be positive")

        effective_val_batch = batch_size if val_batch_size is None else val_batch_size
        effective_val_mc = n_mc_paths if val_mc_paths is None else val_mc_paths

        for step in range(1, steps + 1):
            metrics = self.train_step(batch_size=batch_size, n_mc_paths=n_mc_paths)
            if step_callback is not None:
                step_callback(step, metrics)
            if val_interval > 0 and step % val_interval == 0:
                self.validate(batch_size=effective_val_batch, n_mc_paths=effective_val_mc)
                if self.should_stop_early():
                    break
        return self.history

    def save_checkpoint(self, path: str | Path) -> Path:
        """Persist model, optimizer, scheduler, and history to disk."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": None if self.scheduler is None else self.scheduler.state_dict(),
            "warmup_state": None if self.warmup_scheduler is None else self.warmup_scheduler.state_dict(),
            "global_step": self.global_step,
            "history": self.history.__dict__,
            "model_config": self.model.architecture_dict(),
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> dict[str, object]:
        """Load checkpoint and restore optimizer/scheduler/history state."""
        checkpoint_path = Path(path)
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"], strict=strict)
        self.optimizer.load_state_dict(payload["optimizer_state"])
        if self.scheduler is not None and payload.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        if self.warmup_scheduler is not None and payload.get("warmup_state") is not None:
            self.warmup_scheduler.load_state_dict(payload["warmup_state"])
        self.global_step = int(payload.get("global_step", 0))
        history = payload.get("history")
        if isinstance(history, dict):
            self.history = TrainerHistory(**history)
        return payload


__all__ = [
    "FKProblem",
    "TrainStepMetrics",
    "TrainerHistory",
    "FeynmanKacTrainer",
]
