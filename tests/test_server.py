import pytest
from fastapi.testclient import TestClient

from game2048 import server
from game2048.core import Game


@pytest.fixture
def client():
    server.state.game = Game(seed=1)
    server.state.agents.clear()
    return TestClient(server.app)


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "<title>2048</title>" in r.text


def test_get_state_shape(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"board", "score", "over", "legal_moves", "max_tile"}
    assert len(body["board"]) == 4 and all(len(row) == 4 for row in body["board"])


def test_new_game_with_seed_is_reproducible(client):
    a = client.post("/api/new", json={"seed": 42}).json()
    b = client.post("/api/new", json={"seed": 42}).json()
    assert a["board"] == b["board"]
    assert a["score"] == 0


def test_new_game_without_body(client):
    r = client.post("/api/new")
    assert r.status_code == 200
    assert r.json()["score"] == 0


def test_move_by_int_and_by_name(client):
    server.state.game.board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    r = client.post("/api/move", json={"direction": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["moved"] is True and body["reward"] == 4
    assert body["board"][0][0] == 4

    server.state.game.board = [[0, 0, 2, 2], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    r = client.post("/api/move", json={"direction": "right"})
    assert r.json()["board"][0][3] == 4


def test_move_rejects_bad_direction(client):
    assert client.post("/api/move", json={"direction": 7}).status_code == 422
    assert client.post("/api/move", json={"direction": "sideways"}).status_code == 422
    assert client.post("/api/move", json={}).status_code == 422


def test_invalid_move_returns_moved_false(client):
    server.state.game.board = [[2, 4, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    body = client.post("/api/move", json={"direction": "left"}).json()
    assert body["moved"] is False and body["reward"] == 0


def test_list_agents(client):
    names = client.get("/api/agents").json()
    assert "random" in names and "greedy" in names


def test_agent_step_applies_a_legal_move(client):
    before = client.get("/api/state").json()
    body = client.post("/api/agent/step", json={"agent": "greedy"}).json()
    assert body["moved"] is True
    assert body["direction"] in before["legal_moves"]
    assert body["board"] != before["board"]


def test_agent_step_unknown_agent_is_404(client):
    assert client.post("/api/agent/step", json={"agent": "nope"}).status_code == 404


def test_agent_step_on_finished_game_is_409(client):
    server.state.game.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    assert client.post("/api/agent/step", json={"agent": "random"}).status_code == 409


def test_agent_instances_are_cached_between_requests(client):
    client.post("/api/agent/step", json={"agent": "random"})
    first = server.state.agents["random"]
    client.post("/api/agent/step", json={"agent": "random"})
    assert server.state.agents["random"] is first


def test_nn_agent_without_checkpoint_is_503(client, monkeypatch, tmp_path):
    monkeypatch.setenv("GAME2048_CHECKPOINT", str(tmp_path / "missing.pt"))
    r = client.post("/api/agent/step", json={"agent": "nn"})
    assert r.status_code == 503
    assert "train" in r.json()["detail"]


def test_ntuple_agent_without_weights_is_503(client, monkeypatch, tmp_path):
    monkeypatch.setenv("GAME2048_NTUPLE", str(tmp_path / "missing.bin"))
    r = client.post("/api/agent/step", json={"agent": "ntuple"})
    assert r.status_code == 503
    assert "train" in r.json()["detail"]


def test_new_game_accepts_a_spawn_mode(client):
    body = client.post("/api/new", json={"seed": 1, "mode": "kind"}).json()
    assert body["mode"] == "kind"
    assert client.get("/api/state").json()["mode"] == "kind"
    assert client.post("/api/new", json={"mode": "nasty"}).status_code == 422


def test_modes_endpoint_lists_them(client):
    assert client.get("/api/modes").json() == ["random", "kind", "best", "evil", "choose"]


def test_choose_mode_flow_over_the_api(client):
    assert "choose" in client.get("/api/modes").json()
    s = client.post("/api/new", json={"seed": 1, "mode": "choose"}).json()
    assert s["mode"] == "choose" and s["awaiting_tile"] is False
    server.state.game.board = [[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    s = client.post("/api/move", json={"direction": "left"}).json()
    assert s["moved"] and s["awaiting_tile"] is True and s["legal_moves"] == []
    assert client.post("/api/move", json={"direction": "up"}).status_code == 409
    assert client.post("/api/place", json={"row": 0, "col": 0, "value": 2}).status_code == 409   # occupied
    assert client.post("/api/place", json={"row": 0, "col": 1, "value": 8}).status_code == 422   # not 2/4
    s = client.post("/api/place", json={"row": 0, "col": 1, "value": 4}).json()
    assert s["awaiting_tile"] is False and s["board"][0] == [4, 4, 0, 0]
    assert client.post("/api/place", json={"row": 1, "col": 1, "value": 2}).status_code == 409   # not waiting


def test_agent_step_in_choose_mode_moves_and_places(client):
    client.post("/api/new", json={"seed": 1, "mode": "choose"})
    s = client.post("/api/agent/step", json={"agent": "greedy"}).json()
    assert s["moved"] is True and s["awaiting_tile"] is False
    assert s["placed"]["value"] in (2, 4)
    r, c = s["placed"]["row"], s["placed"]["col"]
    assert s["board"][r][c] == s["placed"]["value"]


def test_agent_run_plays_many_steps_within_a_time_budget(client):
    body = client.post("/api/agent/run", json={"agent": "greedy", "ms": 50}).json()
    assert body["steps"] > 1 and body["score"] > 0 and body["reward"] > 0
    assert body["board"] == client.get("/api/state").json()["board"]


def test_agent_run_respects_max_steps_in_cool_mode(client):
    client.post("/api/new", json={"seed": 1, "mode": "choose"})
    body = client.post("/api/agent/run", json={"agent": "greedy", "ms": 1000, "max_steps": 5}).json()
    assert body["steps"] == 5 and body["awaiting_tile"] is False and body["mode"] == "choose"


def test_agent_run_on_finished_game_is_409(client):
    server.state.game.board = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
    assert client.post("/api/agent/run", json={"agent": "random"}).status_code == 409
