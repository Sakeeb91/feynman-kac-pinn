"""Schemas for simulation lifecycle endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

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
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    lr: Optional[float] = None
    grad_norm: Optional[float] = None
    error: Optional[str] = None


class SimulationResponse(BaseModel):
    id: str
    problem_id: str
    status: SimulationStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime
    updated_at: datetime
    metrics: Optional[SimulationMetrics] = None


class SimulationListResponse(BaseModel):
    items: list[SimulationResponse]


class CancellationResponse(BaseModel):
    id: str
    status: SimulationStatus
