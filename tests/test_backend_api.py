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
