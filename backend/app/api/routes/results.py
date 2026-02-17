"""Result retrieval endpoints for completed and in-progress simulations."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models.result import ResultEnvelope
from backend.app.services.result_cache import result_cache
from backend.app.services.simulation_manager import simulation_manager

router = APIRouter()


@router.get("/{simulation_id}", response_model=ResultEnvelope)
async def get_result(simulation_id: str) -> ResultEnvelope:
    """Return cached result, or derive one from current simulation state."""
    if not simulation_manager.exists(simulation_id):
        raise HTTPException(status_code=404, detail="Simulation not found")

    cached = result_cache.get(simulation_id)
    if cached is not None and cached.progress >= 1.0:
        return ResultEnvelope(item=cached)

    snapshot = simulation_manager.get_raw(simulation_id)
    assert snapshot is not None
    history = simulation_manager.get_history(simulation_id)
    item = result_cache.upsert_from_simulation(snapshot, history)
    return ResultEnvelope(item=item)
