"""Simulation lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from backend.app.models.simulation import (
    CancellationResponse,
    SimulationCreate,
    SimulationListResponse,
    SimulationResponse,
)
from backend.app.services.simulation_manager import simulation_manager

router = APIRouter()


@router.post("", response_model=SimulationResponse, status_code=status.HTTP_201_CREATED)
async def create_simulation(
    request: SimulationCreate,
    background_tasks: BackgroundTasks,
) -> SimulationResponse:
    """Create a simulation and schedule asynchronous execution."""
    simulation = simulation_manager.create(request)
    background_tasks.add_task(simulation_manager.run_simulation, simulation.id)
    return simulation


@router.get("/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(simulation_id: str) -> SimulationResponse:
    simulation = simulation_manager.get(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulation


@router.get("", response_model=SimulationListResponse)
async def list_simulations() -> SimulationListResponse:
    return SimulationListResponse(items=simulation_manager.list_all())


@router.delete("/{simulation_id}", response_model=CancellationResponse)
async def cancel_simulation(simulation_id: str) -> CancellationResponse:
    cancelled = simulation_manager.cancel(simulation_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Simulation not found")
    simulation = simulation_manager.get(simulation_id)
    assert simulation is not None
    return CancellationResponse(id=simulation_id, status=simulation.status)
