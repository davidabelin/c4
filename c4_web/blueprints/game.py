"""Gameplay API routes for human-vs-agent Connect4 interactions."""

from __future__ import annotations

import json
from random import Random
from time import perf_counter

from flask import Blueprint, current_app, jsonify, request

from c4_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from c4_core.engine import play_human_turn, replay_ai_agent_state
from c4_core.types import Connect4Config
from c4_web.runtime import GameRuntimeState


game_bp = Blueprint("game_api", __name__, url_prefix="/api/v1")


def _repo():
    return current_app.extensions["repository"]


def _runtime():
    return current_app.extensions["game_runtime"]


def _config() -> Connect4Config:
    return Connect4Config(rows=6, columns=7, inarow=4)


def _decode_board(raw_value):
    if isinstance(raw_value, list):
        return [int(v) for v in raw_value]
    if isinstance(raw_value, str):
        return [int(v) for v in json.loads(raw_value)]
    return [0] * 42


def _serialize_game(game: dict) -> dict:
    board = _decode_board(game.get("current_board_json", "[]"))
    return {
        "game_id": int(game["id"]),
        "agent_name": game["agent_name"],
        "session_index": int(game["session_index"]),
        "status": game["status"],
        "winner": game["winner"],
        "rounds_played": int(game["rounds_played"]),
        "score_player": int(game["score_player"]),
        "score_ai": int(game["score_ai"]),
        "score_ties": int(game["score_ties"]),
        "board": board,
        "updated_at": game["updated_at"],
    }


def _resolve_agent_factory_and_signature(game: dict):
    if str(game["agent_name"]) == "active_model":
        model_record = _repo().get_active_model()
        if model_record is None:
            raise RuntimeError("No active model is available. Activate a trained model first.")
        model_id = int(model_record["id"])
        artifact_path = str(model_record["artifact_path"])
        return (lambda: ModelBackedAgent(artifact_path), f"active_model:{model_id}")
    name = str(game["agent_name"])
    return (lambda: build_heuristic_agent(name), f"heuristic:{name}")


def _load_runtime_state(game: dict) -> GameRuntimeState:
    game_id = int(game["id"])
    session_index = int(game["session_index"])
    cached = _runtime().get(game_id)
    if cached is not None and cached.session_index == session_index:
        if str(game["agent_name"]) == "active_model":
            if cached.signature.startswith("active_model:"):
                return cached
        else:
            return cached

    agent_factory, signature = _resolve_agent_factory_and_signature(game)
    agent = agent_factory()
    moves = _repo().list_moves(game_id, session_index=session_index)
    replay_ai_agent_state(agent, moves, _config())
    state = GameRuntimeState(
        game_id=game_id,
        agent_name=str(game["agent_name"]),
        session_index=session_index,
        signature=signature,
        agent=agent,
    )
    _runtime().put(state)
    return state


@game_bp.get("/agents")
def list_agents():
    specs = list_agent_specs()
    models = _repo().list_models(limit=1)
    payload = [
        {
            "name": spec.name,
            "description": spec.description,
            "type": "heuristic",
        }
        for spec in specs
    ]
    if models:
        payload.append(
            {
                "name": "active_model",
                "description": "Use the model marked active in the model registry.",
                "type": "trained",
            }
        )
    return jsonify({"agents": payload})


@game_bp.post("/games")
def create_game():
    payload = request.get_json(silent=True) or {}
    agent_name = str(payload.get("agent", current_app.config["DEFAULT_AGENT"]))
    available = {spec.name for spec in list_agent_specs()}
    if agent_name != "active_model" and agent_name not in available:
        return jsonify({"error": f"Unknown agent '{agent_name}'."}), 400
    if agent_name == "active_model" and _repo().get_active_model() is None:
        return jsonify({"error": "No active model is available. Train and activate a model first."}), 400
    game = _repo().create_game(agent_name=agent_name, board=[0] * 42)
    _runtime().forget_game(int(game["id"]))
    return jsonify({"game": _serialize_game(game)}), 201


@game_bp.get("/games/<int:game_id>")
def get_game(game_id: int):
    game = _repo().get_game(game_id)
    if game is None:
        return jsonify({"error": "Game not found."}), 404
    moves = _repo().list_moves(game_id, session_index=int(game["session_index"]))
    return jsonify({"game": _serialize_game(game), "moves": moves})


@game_bp.post("/games/<int:game_id>/move")
def play_move(game_id: int):
    started = perf_counter()
    payload = request.get_json(silent=True) or {}
    if "action" not in payload:
        return jsonify({"error": "Request body must include 'action'."}), 400

    runtime_state = _runtime().get(game_id)
    if runtime_state is None:
        game = _repo().get_game(game_id)
        if game is None:
            return jsonify({"error": "Game not found."}), 404
        if str(game.get("status")) != "active":
            return jsonify({"error": "Game is already complete. Reset or create a new game."}), 409
        try:
            runtime_state = _load_runtime_state(game)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 400
    else:
        game = _repo().get_game(game_id)
        if game is None:
            _runtime().forget_game(game_id)
            return jsonify({"error": "Game not found."}), 404

    board = _decode_board(game.get("current_board_json", "[]"))

    try:
        result = play_human_turn(
            board=board,
            player_action=int(payload["action"]),
            agent=runtime_state.agent,
            config=_config(),
            round_index=int(game["rounds_played"]),
            rng=Random(),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        inserted_moves, updated_game = _repo().record_turn_and_update_game(
            game_id=game_id,
            session_index=int(runtime_state.session_index),
            player_action=int(result.player_action),
            ai_action=(int(result.ai_action) if result.ai_action is not None else None),
            board_before_player=result.board_before_player,
            board_after_player=result.board_after_player,
            board_after_ai=result.board_after_ai,
            outcome=str(result.outcome),
        )
    except KeyError:
        _runtime().forget_game(game_id)
        return jsonify({"error": "Game session changed. Start a new game and try again."}), 409

    if str(updated_game.get("status")) == "completed":
        _runtime().forget_game(game_id)

    return jsonify(
        {
            "game": _serialize_game(updated_game),
            "turn": {
                "round_index": int(result.round_index),
                "player_action": int(result.player_action),
                "ai_action": (int(result.ai_action) if result.ai_action is not None else None),
                "outcome": result.outcome,
                "reward_delta": int(result.reward_delta),
                "board_after_player": result.board_after_player,
                "board_after_ai": result.board_after_ai,
                "server_elapsed_ms": int(round((perf_counter() - started) * 1000)),
            },
            "moves": inserted_moves,
        }
    )


@game_bp.post("/games/<int:game_id>/reset")
def reset_game(game_id: int):
    updated_game = _repo().reset_game(game_id, board=[0] * 42)
    if updated_game is None:
        return jsonify({"error": "Game not found."}), 404
    _runtime().forget_game(game_id)
    return jsonify({"game": _serialize_game(updated_game)})
