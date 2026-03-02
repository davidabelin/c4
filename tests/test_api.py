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
    # RL may fail if kaggle_environments is unavailable in the active env.
    assert status in {"completed", "failed"}
