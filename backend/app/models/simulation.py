"""Schemas for simulation lifecycle endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .common import SimulationStatus


class TrainingConfig(BaseModel):
    n_steps: int = Field(default=25, ge=1, le=100000)
    batch_size: int = Field(default=64, ge=1, le=100000)
    n_mc_paths: int = Field(default=256, ge=1, le=100000)
    learning_rate: float = Field(default=1e-3, gt=0.0)


class SimulationCreate(BaseModel):
    problem_id: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)


class SimulationMetrics(BaseModel):
    loss: float | None = None
    val_loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None
    error: str | None = None


class SimulationResponse(BaseModel):
    id: str
    problem_id: str
    status: SimulationStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    metrics: SimulationMetrics | None = None


class SimulationListResponse(BaseModel):
    items: list[SimulationResponse]


class CancellationResponse(BaseModel):
    id: str
    status: SimulationStatus
