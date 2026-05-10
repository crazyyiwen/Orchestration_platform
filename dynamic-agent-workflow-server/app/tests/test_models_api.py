"""HTTP-level tests for /api/models, /api/providers, /api/models/test."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_list_models_returns_registered_entries() -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    ids = {m["id"] for m in data["models"]}
    assert "mock-fast" in ids
    mock_entry = next(m for m in data["models"] if m["id"] == "mock-fast")
    assert mock_entry["available"] is True
    assert mock_entry["provider"] == "mock"
    assert "chat" in mock_entry["capabilities"]


def test_list_providers_includes_mock_and_marks_availability() -> None:
    with TestClient(create_app()) as client:
        resp = client.get("/api/providers")
    assert resp.status_code == 200
    data = resp.json()
    names = {p["name"] for p in data["providers"]}
    assert "mock" in names
    mock = next(p for p in data["providers"] if p["name"] == "mock")
    assert mock["available"] is True
    assert mock["supports_json_mode"] is True


def test_test_endpoint_runs_against_mock_provider() -> None:
    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/models/test",
            json={"model_id": "mock-fast", "prompt": "ping"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["provider"] == "mock"
    assert data["model"] == "mock-fast"  # tagged with user-facing id
    assert "ping" in (data["content"] or "")
    assert data["usage"]["total_tokens"] > 0


def test_test_endpoint_unknown_model_id_returns_clean_error() -> None:
    with TestClient(create_app()) as client:
        resp = client.post(
            "/api/models/test",
            json={"model_id": "no-such-model", "prompt": "x"},
        )
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "configuration_error"
    assert "unknown model_id" in body["error"]["message"]
