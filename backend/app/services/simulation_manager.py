"""Simulation orchestration service for async training jobs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import RLock
from uuid import uuid4

from backend.app.models.common import SimulationStatus, utc_now
from backend.app.models.simulation import SimulationCreate, SimulationResponse


@dataclass
class SimulationManager:
    """Manages simulation records, status transitions, and cancellation flags."""

    _simulations: dict[str, dict] = field(default_factory=dict)
    _cancel_flags: dict[str, bool] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def create(self, request: SimulationCreate) -> SimulationResponse:
        simulation_id = str(uuid4())
        now = utc_now()
        payload = {
            "id": simulation_id,
            "problem_id": request.problem_id,
            "parameters": request.parameters,
            "training_config": request.training_config.model_dump(),
            "status": SimulationStatus.QUEUED,
            "progress": 0.0,
            "created_at": now,
            "updated_at": now,
            "metrics": None,
        }
        with self._lock:
            self._simulations[simulation_id] = payload
            self._cancel_flags[simulation_id] = False
        return SimulationResponse(**payload)

    def get(self, simulation_id: str) -> SimulationResponse | None:
        with self._lock:
            payload = self._simulations.get(simulation_id)
            if payload is None:
                return None
            return SimulationResponse(**payload)

    def list_all(self) -> list[SimulationResponse]:
        with self._lock:
            return [SimulationResponse(**self._simulations[k]) for k in sorted(self._simulations)]

    def cancel(self, simulation_id: str) -> bool:
        with self._lock:
            if simulation_id not in self._simulations:
                return False
            self._cancel_flags[simulation_id] = True
            payload = self._simulations[simulation_id]
            if payload["status"] in {SimulationStatus.QUEUED, SimulationStatus.RUNNING}:
                payload["status"] = SimulationStatus.CANCELLED
                payload["updated_at"] = utc_now()
            return True

    def exists(self, simulation_id: str) -> bool:
        with self._lock:
            return simulation_id in self._simulations

    def is_cancelled(self, simulation_id: str) -> bool:
        with self._lock:
            return self._cancel_flags.get(simulation_id, False)

    def update_progress(
        self,
        simulation_id: str,
        *,
        progress: float | None = None,
        status: SimulationStatus | None = None,
        metrics: dict | None = None,
    ) -> None:
        with self._lock:
            if simulation_id not in self._simulations:
                return
            payload = self._simulations[simulation_id]
            if progress is not None:
                payload["progress"] = max(0.0, min(1.0, float(progress)))
            if status is not None:
                payload["status"] = status
            if metrics is not None:
                payload["metrics"] = metrics
            payload["updated_at"] = utc_now()

    def get_raw(self, simulation_id: str) -> dict | None:
        with self._lock:
            payload = self._simulations.get(simulation_id)
            return None if payload is None else dict(payload)

    async def run_simulation(self, simulation_id: str) -> None:
        """
        Execute a lightweight async simulation loop.

        This loop is intentionally minimal and gets replaced by full ML execution
        in a later commit; it currently provides non-blocking lifecycle behavior.
        """
        payload = self.get_raw(simulation_id)
        if payload is None:
            return
        total_steps = int(payload["training_config"]["n_steps"])
        self.update_progress(simulation_id, status=SimulationStatus.RUNNING, progress=0.0)
        for step in range(total_steps):
            if self.is_cancelled(simulation_id):
                self.update_progress(simulation_id, status=SimulationStatus.CANCELLED)
                return
            await asyncio.sleep(0)
            progress = (step + 1) / total_steps
            self.update_progress(
                simulation_id,
                progress=progress,
                metrics={"loss": float(total_steps - step) / total_steps},
            )
        self.update_progress(simulation_id, status=SimulationStatus.COMPLETED, progress=1.0)


simulation_manager = SimulationManager()
