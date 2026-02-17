"""Learning-rate scheduler utilities for PINN training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

SchedulerName = Literal["none", "cosine", "plateau"]


@dataclass
class WarmupConfig:
    steps: int = 0
    start_factor: float = 0.1


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Construct cosine annealing scheduler."""
    t_max = max(1, total_steps)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)


def build_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 10,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Construct reduce-on-plateau scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        patience=patience,
        min_lr=min_lr,
    )


def build_linear_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    start_factor: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linearly ramp LR from start_factor to 1.0 over warmup steps."""
    steps = max(1, warmup_steps)
    start = float(start_factor)

    def schedule(step: int) -> float:
        if step >= steps:
            return 1.0
        alpha = step / steps
        return start + (1.0 - start) * alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: SchedulerName,
    total_steps: int,
    min_lr: float = 1e-6,
    plateau_factor: float = 0.5,
    plateau_patience: int = 10,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    """Build scheduler from name and common parameters."""
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return build_cosine_scheduler(optimizer=optimizer, total_steps=total_steps, min_lr=min_lr)
    if scheduler_name == "plateau":
        return build_plateau_scheduler(
            optimizer=optimizer,
            factor=plateau_factor,
            patience=plateau_patience,
            min_lr=min_lr,
        )
    raise ValueError(f"unsupported scheduler name '{scheduler_name}'")


__all__ = [
    "SchedulerName",
    "WarmupConfig",
    "build_scheduler",
    "build_linear_warmup_scheduler",
    "build_cosine_scheduler",
    "build_plateau_scheduler",
]
