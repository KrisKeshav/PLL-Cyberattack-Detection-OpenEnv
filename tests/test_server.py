"""HTTP/WebSocket API contract (OpenEnv runtime validation surface + session-aware simulation)."""

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_openenv_standard_endpoints(client):
    assert client.get("/health").json() == {"status": "healthy"}
    meta = client.get("/metadata").json()
    assert meta["name"] and meta["description"]
    schema = client.get("/schema").json()
    assert {"action", "observation", "state"} <= set(schema)
    assert "attack_detected" in schema["action"]["properties"]
    assert client.post("/mcp", json={}).json()["jsonrpc"] == "2.0"
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/reset", "/step", "/state"} <= set(paths)


def test_reset_accepts_empty_body(client):
    r = client.post("/reset")
    assert r.status_code == 200 and r.json()["observation"]["step"] == 0


def test_full_episode_over_http(client):
    r = client.post("/reset", json={"task_id": 2, "seed": 4, "session_id": "ep"})
    assert r.status_code == 200
    done, steps = False, 0
    while not done:
        r = client.post("/step", json={"action": {"attack_detected": False}, "session_id": "ep"})
        assert r.status_code == 200
        done = r.json()["done"]
        steps += 1
    assert steps == 40
    res = r.json()["observation"]["episode_result"]
    assert 0.0 <= res["grader_score"] <= 1.0
    assert client.get("/state", params={"session_id": "ep"}).json()["scenario"] is not None


def test_sessions_are_isolated(client):
    client.post("/reset", json={"task_id": 0, "seed": 1, "session_id": "a"})
    client.post("/reset", json={"task_id": 1, "seed": 1, "session_id": "b"})
    for _ in range(3):
        client.post("/step", json={"action": {"attack_detected": False}, "session_id": "a"})
    sa = client.get("/state", params={"session_id": "a"}).json()
    sb = client.get("/state", params={"session_id": "b"}).json()
    assert sa["step_count"] == 3 and sb["step_count"] == 0 and sb["task_id"] == 1


def test_flat_action_and_validation_errors(client):
    client.post("/reset", json={"task_id": 0, "seed": 1})
    ok = client.post("/step", json={"attack_detected": True, "attack_type": 1, "confidence": 0.5,
                                    "protective_action": 0})
    assert ok.status_code == 200
    assert client.post("/step", json={"action": {"attack_detected": True, "attack_type": 9}}).status_code == 422
    assert client.post("/step", json={"action": {"attack_detected": True, "bogus": 1}}).status_code == 422
    assert client.post("/step", json={"action": {"attack_detected": False}, "session_id": "zz"}).status_code == 404
    assert client.post("/reset", json={"task_id": 5}).status_code == 422


def test_state_hides_ground_truth_mid_episode(client):
    client.post("/reset", json={"task_id": 2, "seed": 1, "session_id": "gt"})
    client.post("/step", json={"action": {"attack_detected": False}, "session_id": "gt"})
    st = client.get("/state", params={"session_id": "gt"}).json()
    assert st["scenario"] is None and st["grader_score"] is None


def test_websocket_session(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"task_id": 1, "seed": 2}})
        m = ws.receive_json()
        assert m["type"] == "observation" and m["data"]["observation"]["task_id"] == 1
        ws.send_json({"type": "step", "data": {"attack_detected": False}})
        m = ws.receive_json()
        assert m["data"]["observation"]["step"] == 1


def test_tasks_and_demo(client):
    t = client.get("/tasks").json()
    assert len(t["tasks"]) == 3 and t["reward_range"][0] < 0 < t["reward_range"][1]
    d = client.get("/demo", params={"task_id": 0, "seed": 1}).json()
    assert len(d["trace"]) == 40 and 0 <= d["score"] <= 1
    assert client.get("/").status_code == 200
