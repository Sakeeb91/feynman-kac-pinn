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
    _histories: dict[str, dict[str, list[float]]] = field(default_factory=dict)
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
            self._histories[simulation_id] = {
                "train_loss": [],
                "val_loss": [],
                "lr": [],
                "grad_norm": [],
            }
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

    def get_history(self, simulation_id: str) -> dict[str, list[float]] | None:
        with self._lock:
            history = self._histories.get(simulation_id)
            if history is None:
                return None
            return {k: list(v) for k, v in history.items()}

    def set_history(self, simulation_id: str, history: dict[str, list[float]]) -> None:
        with self._lock:
            if simulation_id in self._histories:
                self._histories[simulation_id] = {
                    "train_loss": list(history.get("train_loss", [])),
                    "val_loss": list(history.get("val_loss", [])),
                    "lr": list(history.get("lr", [])),
                    "grad_norm": list(history.get("grad_norm", [])),
                }

    def _run_training_sync(self, simulation_id: str, payload: dict) -> tuple[str, dict | None]:
        from ml.models import FeynmanKacPINN
        from ml.problems import create_problem
        from ml.training import FKProblem, FeynmanKacTrainer

        config = payload["training_config"]
        problem = create_problem(payload["problem_id"], **payload["parameters"])
        model = FeynmanKacPINN(input_dim=problem.dimension)
        trainer = FeynmanKacTrainer(
            model=model,
            problem=FKProblem.from_problem(problem),
            lr=float(config["learning_rate"]),
            max_grad_norm=5.0,
        )

        total_steps = int(config["n_steps"])

        def _step_callback(step: int, metrics) -> None:
            if self.is_cancelled(simulation_id):
                raise RuntimeError("__cancelled__")
            self.update_progress(
                simulation_id,
                progress=step / total_steps,
                metrics={
                    "loss": metrics.loss,
                    "lr": metrics.lr,
                    "grad_norm": metrics.grad_norm,
                },
            )

        try:
            history = trainer.fit(
                steps=total_steps,
                batch_size=int(config["batch_size"]),
                n_mc_paths=int(config["n_mc_paths"]),
                step_callback=_step_callback,
            )
        except RuntimeError as exc:
            if str(exc) == "__cancelled__":
                return "cancelled", None
            raise

        self.set_history(simulation_id, history.__dict__)
        return "completed", trainer.latest_metrics()

    async def run_simulation(self, simulation_id: str) -> None:
        """
        Execute a lightweight async simulation loop.

        This loop is intentionally minimal and gets replaced by full ML execution
        in a later commit; it currently provides non-blocking lifecycle behavior.
        """
        payload = self.get_raw(simulation_id)
        if payload is None:
            return

        self.update_progress(simulation_id, status=SimulationStatus.RUNNING, progress=0.0)
        try:
            outcome, latest_metrics = await asyncio.to_thread(
                self._run_training_sync, simulation_id, payload
            )
            if outcome == "cancelled":
                self.update_progress(simulation_id, status=SimulationStatus.CANCELLED)
            else:
                self.update_progress(
                    simulation_id,
                    status=SimulationStatus.COMPLETED,
                    progress=1.0,
                    metrics=latest_metrics,
                )
        except Exception as exc:  # pragma: no cover - defensive fallback path
            self.update_progress(
                simulation_id,
                status=SimulationStatus.FAILED,
                metrics={"error": str(exc)},
            )


simulation_manager = SimulationManager()
