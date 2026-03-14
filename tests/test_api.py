from __future__ import annotations

import time
from pathlib import Path

import pytest

from c4_web import create_app


@pytest.fixture
def app(tmp_path: Path):
    app = create_app(
        {
            "TESTING": True,
            "DB_PATH": str(tmp_path / "test.db"),
            "EVENTS_DIR": str(tmp_path / "events"),
            "MODELS_DIR": str(tmp_path / "models"),
            "EXPORTS_DIR": str(tmp_path / "exports"),
            "DEFAULT_AGENT": "alpha_beta_v9",
        }
    )
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def _create_game(client, agent: str) -> int:
    response = client.post("/api/v1/games", json={"agent": agent})
    assert response.status_code == 201
    return int(response.get_json()["game"]["game_id"])


def _play_game_until_complete(client, game_id: int, *, max_turns: int = 42) -> None:
    for turn in range(max_turns):
        state = client.get(f"/api/v1/games/{game_id}")
        board = state.get_json()["game"]["board"]
        valid = [col for col in range(7) if board[col] == 0]
        if not valid:
            return
        # Vary actions so AI sees a wider context distribution in training rows.
        action = valid[turn % len(valid)]
        response = client.post(f"/api/v1/games/{game_id}/move", json={"action": action})
        assert response.status_code == 200
        if response.get_json()["game"]["status"] == "completed":
            return


def test_create_move_and_reset_flow(client):
    game_id = _create_game(client, "alpha_beta_v9")
    move = client.post(f"/api/v1/games/{game_id}/move", json={"action": 3})
    assert move.status_code == 200
    payload = move.get_json()
    assert "turn" in payload
    assert payload["turn"]["player_action"] == 3

    state = client.get(f"/api/v1/games/{game_id}")
    assert state.status_code == 200

    reset = client.post(f"/api/v1/games/{game_id}/reset")
    assert reset.status_code == 200
    assert reset.get_json()["game"]["rounds_played"] == 0


def test_game_analysis_returns_column_forecasts(client):
    game_id = _create_game(client, "alpha_beta_v9")
    response = client.get(f"/api/v1/games/{game_id}/analysis?lookahead=4&samples=8")
    assert response.status_code == 200
    analysis = response.get_json()["analysis"]
    assert analysis["lookahead"] == 4
    assert analysis["samples"] == 8
    assert len(analysis["forecasts"]) == 7
    assert analysis["recommended_column"] in range(7)
    assert all("win_estimate" in entry for entry in analysis["forecasts"])


def test_agent_vs_agent_match_endpoint_returns_trace(client):
    response = client.post(
        "/api/v1/matches",
        json={
            "agent_a": "alpha_beta_v9",
            "agent_b": "adaptive_midrange",
            "starting_agent": "agent_a",
            "seed": 7,
        },
    )
    assert response.status_code == 200
    match = response.get_json()["match"]
    assert match["mode"] == "agent_vs_agent"
    assert match["starting_agent"] == "agent_a"
    assert match["mark_agent_a"] == 1
    assert match["mark_agent_b"] == 2
    assert match["moves_played"] >= 1
    assert len(match["trace"]) == int(match["moves_played"])
    assert len(match["final_board"]) == 42
    assert match["winner"] in {"agent_a", "agent_b", "tie"}
    assert match["status"] in {"completed", "truncated"}
    first_move = match["trace"][0]
    assert first_move["actor"] == "agent_a"
    assert len(first_move["board_after"]) == 42


def test_arena_match_job_lifecycle_persists_trace(client):
    create = client.post(
        "/api/v1/arena/matches",
        json={
            "agent_a": "alpha_beta_v9",
            "agent_b": "adaptive_midrange",
            "starting_agent": "agent_a",
            "seed": 7,
        },
    )
    assert create.status_code == 202
    match_id = int(create.get_json()["match"]["id"])

    status = None
    trace = None
    for _ in range(120):
        poll = client.get(f"/api/v1/arena/matches/{match_id}")
        assert poll.status_code == 200
        match = poll.get_json()["match"]
        status = match["status"]
        trace = match["trace"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status == "completed"
    assert trace is not None
    assert len(trace) >= 1
    assert trace[0]["actor"] == "agent_a"


def test_ai_opening_and_undo_flow(client):
    response = client.post("/api/v1/games", json={"agent": "alpha_beta_v9", "opening_player": "ai"})
    assert response.status_code == 201
    payload = response.get_json()
    game = payload["game"]
    board = game["board"]
    assert payload["opening_player"] == "ai"
    assert sum(1 for value in board if int(value) == 2) == 1
    assert int(game["rounds_played"]) == 0

    undo = client.post(f"/api/v1/games/{int(game['game_id'])}/undo")
    assert undo.status_code == 200
    undo_payload = undo.get_json()
    assert undo_payload["undo"]["kind"] == "opening_ai_move"
    assert sum(int(value) for value in undo_payload["game"]["board"]) == 0
    assert int(undo_payload["game"]["rounds_played"]) == 0


def test_undo_after_player_turn(client):
    game_id = _create_game(client, "alpha_beta_v9")
    move = client.post(f"/api/v1/games/{game_id}/move", json={"action": 3})
    assert move.status_code == 200
    moved_state = move.get_json()["game"]
    assert int(moved_state["rounds_played"]) == 1

    undo = client.post(f"/api/v1/games/{game_id}/undo")
    assert undo.status_code == 200
    payload = undo.get_json()
    assert payload["undo"]["kind"] == "turn"
    assert int(payload["game"]["rounds_played"]) == 0
    assert sum(int(value) for value in payload["game"]["board"]) == 0


def test_training_job_lifecycle_and_model_activation(client):
    readiness = None
    for _ in range(8):
        game_id = _create_game(client, "alpha_beta_v9")
        _play_game_until_complete(client, game_id, max_turns=36)
        readiness_response = client.get("/api/v1/training/readiness?lookback=2")
        assert readiness_response.status_code == 200
        readiness = readiness_response.get_json()["readiness"]
        if readiness["can_train"]:
            break
    assert readiness is not None
    assert readiness["can_train"] is True

    create_job = client.post(
        "/api/v1/training/jobs",
        json={
            "model_type": "frequency",
            "lookback": 2,
            "test_size": 0.25,
            "learning_rate": 0.001,
            "hidden_layer_sizes": [32, 16],
            "epochs": 50,
            "random_state": 12,
        },
    )
    assert create_job.status_code == 202
    job_id = int(create_job.get_json()["job"]["id"])

    status = None
    for _ in range(80):
        poll = client.get(f"/api/v1/training/jobs/{job_id}")
        assert poll.status_code == 200
        status = poll.get_json()["job"]["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert status == "completed"

    models_response = client.get("/api/v1/models")
    assert models_response.status_code == 200
    models = models_response.get_json()["models"]
    assert models
    model_id = int(models[0]["id"])

    activate = client.post(f"/api/v1/models/{model_id}/activate")
    assert activate.status_code == 200
    assert activate.get_json()["model"]["is_active"] is True


def test_training_session_curation_supports_gameplay_and_arena_filters(client):
    game_id = _create_game(client, "alpha_beta_v9")
    _play_game_until_complete(client, game_id, max_turns=18)

    arena_create = client.post(
        "/api/v1/arena/matches",
        json={
            "agent_a": "alpha_beta_v9",
            "agent_b": "adaptive_midrange",
            "starting_agent": "agent_a",
            "seed": 3,
        },
    )
    assert arena_create.status_code == 202
    arena_match_id = int(arena_create.get_json()["match"]["id"])

    arena_status = None
    for _ in range(120):
        arena_poll = client.get(f"/api/v1/arena/matches/{arena_match_id}")
        assert arena_poll.status_code == 200
        arena_status = arena_poll.get_json()["match"]["status"]
        if arena_status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert arena_status == "completed"

    sessions_response = client.get("/api/v1/training/sessions")
    assert sessions_response.status_code == 200
    sessions = sessions_response.get_json()["sessions"]
    assert any(session["source_kind"] == "game" for session in sessions)
    assert any(session["source_kind"] == "arena" for session in sessions)

    game_session = next(session for session in sessions if session["source_kind"] == "game")
    arena_session = next(session for session in sessions if session["source_kind"] == "arena")

    include_game = client.post(
        "/api/v1/training/sessions/selection",
        json={
            "source_kind": "game",
            "source_id": game_session["source_id"],
            "session_index": game_session["session_index"],
            "selection": "include",
        },
    )
    assert include_game.status_code == 200

    include_arena = client.post(
        "/api/v1/training/sessions/selection",
        json={
            "source_kind": "arena",
            "source_id": arena_session["source_id"],
            "session_index": arena_session["session_index"],
            "selection": "include",
        },
    )
    assert include_arena.status_code == 200

    human_readiness = client.get("/api/v1/training/readiness?lookback=1&selection_mode=selected&actor_scope=human")
    assert human_readiness.status_code == 200
    human_info = human_readiness.get_json()["readiness"]
    assert human_info["session_count"] == 1
    assert human_info["total_move_rows"] >= 1

    algorithm_readiness = client.get("/api/v1/training/readiness?lookback=1&selection_mode=selected&actor_scope=algorithm")
    assert algorithm_readiness.status_code == 200
    algorithm_info = algorithm_readiness.get_json()["readiness"]
    assert algorithm_info["session_count"] == 2
    assert algorithm_info["total_move_rows"] >= human_info["total_move_rows"]

    reset_arena = client.post(
        "/api/v1/training/sessions/selection",
        json={
            "source_kind": "arena",
            "source_id": arena_session["source_id"],
            "session_index": arena_session["session_index"],
            "selection": None,
        },
    )
    assert reset_arena.status_code == 200


def test_rl_job_lifecycle_creates_model(client):
    create_job = client.post(
        "/api/v1/rl/jobs",
        json={
            "episodes": 3,
            "opponent": "random",
            "seed": 5,
        },
    )
    assert create_job.status_code == 202
    job_id = int(create_job.get_json()["job"]["id"])

    status = None
    for _ in range(120):
        poll = client.get(f"/api/v1/rl/jobs/{job_id}")
        assert poll.status_code == 200
        status = poll.get_json()["job"]["status"]
        if status in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert status == "completed"
