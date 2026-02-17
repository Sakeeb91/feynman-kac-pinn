"""Schemas for simulation result retrieval endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from .common import ResultStatus, SimulationStatus
from .simulation import SimulationMetrics


class TrainingHistory(BaseModel):
    train_loss: list[float] = Field(default_factory=list)
    val_loss: list[float] = Field(default_factory=list)
    lr: list[float] = Field(default_factory=list)
    grad_norm: list[float] = Field(default_factory=list)


class SimulationResult(BaseModel):
    simulation_id: str
    problem_id: str
    simulation_status: SimulationStatus
    result_status: ResultStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    metrics: Optional[SimulationMetrics] = None
    training_history: TrainingHistory = Field(default_factory=TrainingHistory)
    visualization: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime


class ResultEnvelope(BaseModel):
    item: SimulationResult
