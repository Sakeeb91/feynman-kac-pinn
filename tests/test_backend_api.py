import pytest


def test_health_endpoint(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_problems_list_endpoint(api_client) -> None:
    response = api_client.get("/api/v1/problems")
    assert response.status_code == 200
    payload = response.json()
    assert "items" in payload
    assert len(payload["items"]) >= 2


def test_problem_detail_endpoint(api_client) -> None:
    response = api_client.get("/api/v1/problems/black_scholes")
    assert response.status_code == 200
    item = response.json()["item"]
    assert item["id"] == "black_scholes"
    assert any(param["name"] == "dim" for param in item["parameter_schema"])


def test_problem_detail_not_found(api_client) -> None:
    response = api_client.get("/api/v1/problems/does_not_exist")
    assert response.status_code == 404


def test_simulation_lifecycle_and_results(api_client, monkeypatch) -> None:
    from backend.app.models.common import SimulationStatus
    from backend.app.services.simulation_manager import simulation_manager

    async def _fake_run(simulation_id: str) -> None:
        simulation_manager.update_progress(
            simulation_id,
            status=SimulationStatus.RUNNING,
            progress=0.5,
            metrics={"loss": 0.4, "lr": 1e-3, "grad_norm": 0.2},
        )
        simulation_manager.set_history(
            simulation_id,
            {"train_loss": [1.0, 0.4], "val_loss": [], "lr": [1e-3, 1e-3], "grad_norm": [1.2, 0.2]},
        )
        simulation_manager.update_progress(
            simulation_id,
            status=SimulationStatus.COMPLETED,
            progress=1.0,
            metrics={"loss": 0.4, "lr": 1e-3, "grad_norm": 0.2},
        )

    monkeypatch.setattr(simulation_manager, "run_simulation", _fake_run)
    create_response = api_client.post(
        "/api/v1/simulations",
        json={"problem_id": "harmonic_oscillator", "parameters": {"dim": 2}},
    )
    assert create_response.status_code == 201
    simulation_id = create_response.json()["id"]

    status_response = api_client.get(f"/api/v1/simulations/{simulation_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"

    list_response = api_client.get("/api/v1/simulations")
    assert list_response.status_code == 200
    assert len(list_response.json()["items"]) == 1

    result_response = api_client.get(f"/api/v1/results/{simulation_id}")
    assert result_response.status_code == 200
    item = result_response.json()["item"]
    assert item["simulation_status"] == "completed"
    assert item["result_status"] == "ready"
    assert item["training_history"]["train_loss"] == [1.0, 0.4]


def test_simulation_cancel_and_result_not_found(api_client, monkeypatch) -> None:
    from backend.app.services.simulation_manager import simulation_manager

    async def _noop_run(simulation_id: str) -> None:
        del simulation_id
        return

    monkeypatch.setattr(simulation_manager, "run_simulation", _noop_run)
    create_response = api_client.post("/api/v1/simulations", json={"problem_id": "black_scholes"})
    simulation_id = create_response.json()["id"]

    cancel_response = api_client.delete(f"/api/v1/simulations/{simulation_id}")
    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"

    missing_result = api_client.get("/api/v1/results/does-not-exist")
    assert missing_result.status_code == 404
