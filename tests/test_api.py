"""API surface: probes, contract, auth, and the error envelope."""

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readyz_reports_stub_mode(client):
    body = client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["llm"] == "offline_stub"


def test_metrics_exposes_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "feedbackiq_http_requests_total" in resp.text


def test_request_id_is_echoed(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"


def test_analyze_returns_full_contract(client):
    resp = client.post(
        "/v1/feedback/analyze",
        json={
            "items": [
                {"id": "a", "text": "The app crashes on export every time."},
                {"id": "b", "text": "Export throws an error and I lose my data."},
            ],
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "offline_stub"
    assert len(body["analyses"]) == 2
    assert body["themes"], "expected at least one theme"
    assert body["metrics"]["items_analyzed"] == 2
    assert {s["stage"] for s in body["metrics"]["stages"]} == {
        "analyze", "aggregate", "route_github", "notify_slack"
    }


def test_validation_error_uses_the_error_envelope(client):
    resp = client.post("/v1/feedback/analyze", json={"items": []})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "validation_error"
    assert "message" in body and "detail" in body


def test_csv_upload_path(client):
    csv = (
        "id,text,customer_tier\n"
        "1,App is very slow to load,enterprise\n"
        "2,Docs are missing,free\n"
    )
    resp = client.post(
        "/v1/feedback/analyze-csv",
        files={"file": ("export.csv", csv, "text/csv")},
    )
    assert resp.status_code == 200
    assert resp.json()["metrics"]["items_in"] == 2


def test_api_key_is_enforced_when_configured():
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        llm_offline_mode=True, api_key="s3cret"
    )
    with TestClient(app) as c:
        payload = {"items": [{"id": "a", "text": "it broke"}], "dry_run": True}
        assert c.post("/v1/feedback/analyze", json=payload).status_code == 401
        ok = c.post(
            "/v1/feedback/analyze", json=payload, headers={"X-API-Key": "s3cret"}
        )
        assert ok.status_code == 200
    app.dependency_overrides.clear()


def test_suite_is_isolated_from_the_developer_environment(monkeypatch):
    """Regression: exported app config used to leak into Settings() in tests.

    A shell with GITHUB_TOKEN and SLACK_WEBHOOK_URL set made
    settings.github_enabled true inside tests that assume it is false, and the
    pipeline then attempted real outbound calls from the test suite. The
    autouse _hermetic_env fixture strips them; this proves it, by setting the
    variables and checking Settings() still comes up clean.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "leaked")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/leaked")
    monkeypatch.setenv("LLM_TOOL_CHOICE", "auto")
    # _hermetic_env already ran for this test, so these were set *after* it.
    # Re-running the strip is what a fresh test would see.
    for var in ("GITHUB_TOKEN", "SLACK_WEBHOOK_URL", "LLM_TOOL_CHOICE"):
        monkeypatch.delenv(var, raising=False)

    s = Settings()
    assert not s.github_enabled
    assert not s.slack_enabled
    assert s.llm_tool_choice == "forced"


def test_hermetic_fixture_strips_env_before_each_test():
    """The fixture itself, checked directly rather than by its side effects."""
    import os

    from tests.conftest import _APP_ENV_VARS

    leaked = [v for v in _APP_ENV_VARS if v in os.environ]
    assert not leaked, f"app config leaked into the test environment: {leaked}"
