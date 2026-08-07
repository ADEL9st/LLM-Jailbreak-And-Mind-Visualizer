"""HTTP + WebSocket surface. Uses the mock adapter so nothing loads a model."""

import json

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="needs fastapi's TestClient")
TestClient = fastapi_testclient.TestClient

from app import experiments  # noqa: E402
from app.main import BUSY_MESSAGE, app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path / "experiments")
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["boot_id"]


def test_models_lists_the_mock_adapter(client):
    models = client.get("/models").json()
    assert any(item["adapter"] == "mock" for item in models)


# ── experiments API ──────────────────────────────────────────────────────────

def _save(client, **overrides):
    payload = {
        "kind": "run",
        "config": {"prompt": "p", "adapter": "mock", "api_key": "sk-SECRET"},
        "result": {"refused": False},
        "telemetry": {},
        "rows": [],
    }
    payload.update(overrides)
    response = client.post("/experiments", json=payload)
    assert response.status_code == 200
    return response.json()


def test_save_list_get_delete_cycle(client):
    saved = _save(client)
    assert saved["id"]

    listing = client.get("/experiments").json()
    assert [item["id"] for item in listing] == [saved["id"]]

    fetched = client.get(f"/experiments/{saved['id']}").json()
    assert fetched["config"]["prompt"] == "p"

    assert client.delete(f"/experiments/{saved['id']}").json() == {"deleted": saved["id"]}
    assert client.get("/experiments").json() == []


def test_server_strips_the_api_key_even_if_the_client_sends_it(client):
    saved = _save(client)
    assert "api_key" not in saved["config"]


def test_missing_experiment_is_404(client):
    assert client.get("/experiments/20260101-000000-aaaaaa").status_code == 404
    assert client.delete("/experiments/20260101-000000-aaaaaa").status_code == 404


def test_malformed_id_is_rejected_not_500(client):
    # A dot-segment id that survives URL routing must come back as a client
    # error from the id validator, never a traceback.
    response = client.get("/experiments/not..a..valid..id")
    assert response.status_code in (400, 404)


def test_csv_export_for_a_benchmark_report(client):
    saved = _save(client, kind="benchmark", rows=[{
        "id": "b-001", "category": "harm", "prompt": "p", "expected_refusal": True,
        "refused": True, "verdict": "PASS", "peak": 0.8, "state": "refusal_locked",
        "elapsed": 1.0, "text": "no",
    }])
    response = client.get(f"/experiments/{saved['id']}/csv")
    assert response.status_code == 200
    assert "text/csv" in response.headers["content-type"]
    assert f"{saved['id']}.csv" in response.headers["content-disposition"]
    lines = response.text.strip().splitlines()
    assert lines[0].startswith("id,category,prompt")
    assert lines[1].startswith("b-001,harm")


def test_csv_export_of_a_rowless_run_is_a_400(client):
    saved = _save(client)
    assert client.get(f"/experiments/{saved['id']}/csv").status_code == 400


def test_manual_review_endpoint_updates_the_report(client):
    saved = _save(client, result={"refused": False, "assessment": {"category": "truncated"}})
    response = client.patch(f"/experiments/{saved['id']}/review", json={
        "review": {
            "verdict": "inconclusive",
            "category": "truncated",
            "technical_accuracy": None,
            "notes": "Stopped at the token cap.",
            "reviewer": "Burak",
        }
    })
    assert response.status_code == 200
    assert response.json()["review"]["verdict"] == "inconclusive"
    listing = client.get("/experiments").json()
    assert listing[0]["review_verdict"] == "inconclusive"


# ── /ws/run ──────────────────────────────────────────────────────────────────

MOCK_REQUEST = {"prompt": "hello", "adapter": "mock", "model": "mock-qwen2.5-1.5b", "max_new_tokens": 4}


def test_ws_streams_a_mock_run(client):
    with client.websocket_connect("/ws/run") as socket:
        socket.send_text(json.dumps(MOCK_REQUEST))
        types = []
        while True:
            try:
                types.append(json.loads(socket.receive_text())["type"])
            except Exception:
                break
    assert "run_started" in types
    assert "token" in types
    assert "run_completed" in types
    assert "error" not in types


def test_ws_stop_message_cooperatively_cancels_a_run(client):
    request = {**MOCK_REQUEST, "max_new_tokens": 1000, "token_limit_mode": "fixed"}
    with client.websocket_connect("/ws/run") as socket:
        socket.send_text(json.dumps(request))
        assert json.loads(socket.receive_text())["type"] == "run_started"
        socket.send_text(json.dumps({"type": "stop"}))
        completed = None
        while True:
            try:
                item = json.loads(socket.receive_text())
            except Exception:
                break
            if item["type"] == "run_completed":
                completed = item["data"]
    assert completed is not None
    assert completed["finish_reason"] == "cancelled"


def test_ws_reports_an_invalid_request_as_an_error_event(client):
    with client.websocket_connect("/ws/run") as socket:
        socket.send_text(json.dumps({"prompt": ""}))  # fails min_length
        event = json.loads(socket.receive_text())
    assert event["type"] == "error"


def test_second_concurrent_run_is_rejected_with_a_clear_message(client):
    """Adapters are shared singletons — a second run would clobber the first
    one's hooks, so the server must refuse it rather than interleave."""
    with client.websocket_connect("/ws/run") as first:
        first.send_text(json.dumps(MOCK_REQUEST))
        # Wait until the first run actually holds the lock.
        assert json.loads(first.receive_text())["type"] == "run_started"

        with client.websocket_connect("/ws/run") as second:
            second.send_text(json.dumps(MOCK_REQUEST))
            event = json.loads(second.receive_text())

        assert event["type"] == "error"
        assert event["data"]["message"] == BUSY_MESSAGE


def test_disconnecting_mid_run_releases_the_lock(client):
    """If the lock leaked on an abandoned run, the server would be wedged until
    restart — the most damaging way this could fail."""
    with client.websocket_connect("/ws/run") as socket:
        socket.send_text(json.dumps(MOCK_REQUEST))
        socket.receive_text()  # first frame, then walk away mid-generation

    with client.websocket_connect("/ws/run") as socket:
        socket.send_text(json.dumps(MOCK_REQUEST))
        event = json.loads(socket.receive_text())
    assert event["type"] != "error", "lock was not released after a disconnect"


def test_the_lock_is_released_so_a_later_run_still_works(client):
    for _ in range(2):
        with client.websocket_connect("/ws/run") as socket:
            socket.send_text(json.dumps(MOCK_REQUEST))
            types = []
            while True:
                try:
                    types.append(json.loads(socket.receive_text())["type"])
                except Exception:
                    break
        assert "run_completed" in types
