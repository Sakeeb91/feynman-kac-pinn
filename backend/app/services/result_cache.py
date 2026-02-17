"""In-memory result cache for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from backend.app.models.common import ResultStatus, SimulationStatus, utc_now
from backend.app.models.result import SimulationResult, TrainingHistory
from backend.app.models.simulation import SimulationMetrics


@dataclass
class ResultCache:
    """Thread-safe cache for simulation result payloads."""

    _results: dict[str, SimulationResult] = field(default_factory=dict)
    _lock: RLock = field(default_factory=RLock)

    def get(self, simulation_id: str) -> SimulationResult | None:
        with self._lock:
            return self._results.get(simulation_id)

    def set(self, result: SimulationResult) -> None:
        with self._lock:
            self._results[result.simulation_id] = result

    def delete(self, simulation_id: str) -> bool:
        with self._lock:
            if simulation_id not in self._results:
                return False
            del self._results[simulation_id]
            return True

    def list_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._results.keys())

    def list_all(self) -> list[SimulationResult]:
        with self._lock:
            return [self._results[key] for key in sorted(self._results)]

    def clear(self) -> None:
        with self._lock:
            self._results.clear()

    def upsert_from_simulation(
        self,
        simulation_payload: dict,
        history_payload: dict[str, list[float]] | None,
    ) -> SimulationResult:
        status = simulation_payload["status"]
        if status == SimulationStatus.COMPLETED:
            result_status = ResultStatus.READY
        elif status in {SimulationStatus.RUNNING, SimulationStatus.QUEUED}:
            result_status = ResultStatus.PARTIAL
        else:
            result_status = ResultStatus.NOT_READY

        metrics = simulation_payload.get("metrics") or None
        result = SimulationResult(
            simulation_id=simulation_payload["id"],
            problem_id=simulation_payload["problem_id"],
            simulation_status=status,
            result_status=result_status,
            progress=simulation_payload.get("progress", 0.0),
            metrics=None if metrics is None else SimulationMetrics(**metrics),
            training_history=TrainingHistory(**(history_payload or {})),
            visualization={"progress_curve": history_payload.get("train_loss", []) if history_payload else []},
            updated_at=simulation_payload.get("updated_at", utc_now()),
        )
        self.set(result)
        return result


result_cache = ResultCache()
