"""Shared schema primitives for backend models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SimulationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResultStatus(str, Enum):
    NOT_READY = "not_ready"
    PARTIAL = "partial"
    READY = "ready"
