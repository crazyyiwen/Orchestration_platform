from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "env" in data
    assert data["version"] == "0.1.0"
    # mongo may be "ok" or "unavailable" depending on local env;
    # the field must exist and be one of the documented values.
    assert data["mongo"] in {"ok", "unavailable", "unconfigured"}


def test_settings_redaction_in_security_helper() -> None:
    from app.core.security import sanitize_error

    redacted = sanitize_error(
        {
            "error": {
                "message": "boom sk-ant-abcdef1234567890XYZ tail",
                "details": {"api_key": "sk-1234567890abcdef", "nested": {"token": "xyz"}},
            }
        }
    )
    assert "sk-ant" not in redacted["error"]["message"]
    assert redacted["error"]["details"]["api_key"] == "***"
    assert redacted["error"]["details"]["nested"]["token"] == "***"
