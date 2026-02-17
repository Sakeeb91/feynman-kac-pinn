"""Simulation lifecycle endpoints."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, status

from backend.app.models.simulation import SimulationCreate, SimulationResponse
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
