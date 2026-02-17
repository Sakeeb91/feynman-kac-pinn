"""Pytest configuration for project-local imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _reset_backend_state() -> None:
    from backend.app.services.result_cache import result_cache
    from backend.app.services.simulation_manager import simulation_manager

    simulation_manager.clear()
    result_cache.clear()
    yield
    simulation_manager.clear()
    result_cache.clear()


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as client:
        yield client
