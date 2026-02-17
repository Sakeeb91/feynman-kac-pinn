"""In-memory result cache for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from backend.app.models.result import SimulationResult


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


result_cache = ResultCache()
