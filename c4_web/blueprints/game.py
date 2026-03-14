"""Gameplay API routes for human-vs-agent Connect4 interactions."""

from __future__ import annotations

import json
import math
import random as pyrandom
from random import Random
from time import perf_counter

from flask import Blueprint, current_app, jsonify, request

from c4_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from c4_core.board import board_to_grid, drop_piece, has_any_four, valid_columns
from c4_core.engine import play_human_turn, replay_ai_agent_state, select_ai_action
from c4_core.matches import play_agent_match
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


def _available_agent_names() -> list[str]:
    return [spec.name for spec in list_agent_specs()]


def _default_match_opponent(agent_name: str) -> str:
    for candidate in _available_agent_names():
        if candidate != agent_name:
            return candidate
    return agent_name


def _build_agent_from_name(agent_name: str):
    if agent_name == "active_model":
        model_record = _repo().get_active_model()
        if model_record is None:
            raise RuntimeError("No active model is available. Activate a trained model first.")
        return ModelBackedAgent(str(model_record["artifact_path"]))
    available = set(_available_agent_names())
    if agent_name not in available:
        raise KeyError(f"Unknown agent '{agent_name}'.")
    return build_heuristic_agent(agent_name)


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


def _terminal_outcome_for_board(board: list[int]) -> str | None:
    grid = board_to_grid(board, _config())
    if has_any_four(grid, mark=1, config=_config()):
        return "player"
    if has_any_four(grid, mark=2, config=_config()):
        return "ai"
    if not valid_columns(board, _config()):
        return "tie"
    return None


def _window_score(window: list[int], perspective_mark: int, inarow: int) -> float:
    opponent_mark = 1 if perspective_mark == 2 else 2
    perspective_count = window.count(perspective_mark)
    opponent_count = window.count(opponent_mark)
    empty_count = window.count(0)
    if perspective_count and opponent_count:
        return 0.0
    if perspective_count == inarow:
        return 500.0
    if opponent_count == inarow:
        return -500.0
    if perspective_count == inarow - 1 and empty_count == 1:
        return 36.0
    if opponent_count == inarow - 1 and empty_count == 1:
        return -44.0
    if perspective_count == inarow - 2 and empty_count == 2:
        return 8.0
    if opponent_count == inarow - 2 and empty_count == 2:
        return -10.0
    return 0.0


def _heuristic_probability(board: list[int], perspective_mark: int) -> float:
    config = _config()
    grid = board_to_grid(board, config)
    score = 0.0

    center_col = int(config.columns) // 2
    center_values = [int(grid[row, center_col]) for row in range(int(config.rows))]
    score += 3.0 * center_values.count(int(perspective_mark))
    score -= 3.0 * center_values.count(1 if perspective_mark == 2 else 2)

    for row in range(int(config.rows)):
        for col in range(int(config.columns) - int(config.inarow) + 1):
            score += _window_score(list(grid[row, col : col + int(config.inarow)]), perspective_mark, int(config.inarow))
    for row in range(int(config.rows) - int(config.inarow) + 1):
        for col in range(int(config.columns)):
            score += _window_score(list(grid[row : row + int(config.inarow), col]), perspective_mark, int(config.inarow))
    for row in range(int(config.rows) - int(config.inarow) + 1):
        for col in range(int(config.columns) - int(config.inarow) + 1):
            score += _window_score(
                [int(grid[row + step, col + step]) for step in range(int(config.inarow))],
                perspective_mark,
                int(config.inarow),
            )
    for row in range(int(config.inarow) - 1, int(config.rows)):
        for col in range(int(config.columns) - int(config.inarow) + 1):
            score += _window_score(
                [int(grid[row - step, col + step]) for step in range(int(config.inarow))],
                perspective_mark,
                int(config.inarow),
            )

    return float(1.0 / (1.0 + math.exp(-score / 28.0)))


def _outcome_to_probability(outcome: str, perspective_mark: int) -> float:
    if outcome == "tie":
        return 0.5
    if outcome == "player":
        return 1.0 if perspective_mark == 1 else 0.0
    if outcome == "ai":
        return 1.0 if perspective_mark == 2 else 0.0
    return 0.5


def _simulate_guided_outcome(
    board: list[int],
    *,
    next_mark: int,
    perspective_mark: int,
    lookahead: int,
    sample_index: int,
) -> float:
    working = list(board)
    current_mark = int(next_mark)
    seed_value = 1009 + sample_index * 131 + lookahead * 17 + sum((idx + 1) * int(value) for idx, value in enumerate(board))
    saved_state = pyrandom.getstate()
    pyrandom.seed(seed_value)
    try:
        for _ in range(max(0, int(lookahead))):
            terminal = _terminal_outcome_for_board(working)
            if terminal is not None:
                return _outcome_to_probability(terminal, perspective_mark)
            agent = build_heuristic_agent("alpha_beta_v9")
            action = select_ai_action(agent, working, config=_config(), mark=current_mark, rng=Random(seed_value))
            next_grid = drop_piece(board_to_grid(working, _config()), action, mark=current_mark, config=_config())
            working = next_grid.reshape(-1).astype(int).tolist()
            current_mark = 1 if current_mark == 2 else 2
    finally:
        pyrandom.setstate(saved_state)

    terminal = _terminal_outcome_for_board(working)
    if terminal is not None:
        return _outcome_to_probability(terminal, perspective_mark)
    return _heuristic_probability(working, perspective_mark)


def _forecast_columns(board: list[int], lookahead: int, samples: int) -> list[dict]:
    forecasts: list[dict] = []
    config = _config()
    for column in valid_columns(board, config):
        next_grid = drop_piece(board_to_grid(board, config), column, mark=1, config=config)
        board_after_player = next_grid.reshape(-1).astype(int).tolist()
        immediate = _terminal_outcome_for_board(board_after_player)
        if immediate is not None:
            win_estimate = _outcome_to_probability(immediate, 1)
        else:
            estimates = [
                _simulate_guided_outcome(
                    board_after_player,
                    next_mark=2,
                    perspective_mark=1,
                    lookahead=max(0, int(lookahead) - 1),
                    sample_index=sample_index,
                )
                for sample_index in range(max(1, int(samples)))
            ]
            win_estimate = float(sum(estimates) / len(estimates))
        forecasts.append(
            {
                "column": int(column),
                "win_estimate": round(float(win_estimate), 4),
                "label": f"{round(float(win_estimate) * 100)}%",
            }
        )
    forecasts.sort(key=lambda item: (-float(item["win_estimate"]), int(item["column"])))
    return forecasts


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
    opening_player = str(payload.get("opening_player", "player")).strip().lower()
    if opening_player not in {"player", "ai", "random"}:
        return jsonify({"error": "opening_player must be one of: player, ai, random."}), 400

    available = set(_available_agent_names())
    if agent_name != "active_model" and agent_name not in available:
        return jsonify({"error": f"Unknown agent '{agent_name}'."}), 400
    if agent_name == "active_model" and _repo().get_active_model() is None:
        return jsonify({"error": "No active model is available. Train and activate a model first."}), 400
    if opening_player == "random":
        opening_player = "player" if Random().random() < 0.5 else "ai"

    game = _repo().create_game(agent_name=agent_name, board=[0] * 42)
    _runtime().forget_game(int(game["id"]))

    opening_move = None
    if opening_player == "ai":
        try:
            runtime_state = _load_runtime_state(game)
            board_before = [0] * 42
            ai_action = select_ai_action(
                runtime_state.agent,
                board_before,
                config=_config(),
                mark=2,
                rng=Random(),
            )
            next_grid = drop_piece(board_to_grid(board_before, _config()), ai_action, mark=2, config=_config())
            board_after = next_grid.reshape(-1).astype(int).tolist()
            stored_move, game = _repo().record_ai_opening_move(
                game_id=int(game["id"]),
                session_index=int(game["session_index"]),
                ai_action=ai_action,
                board_before=board_before,
                board_after=board_after,
            )
            opening_move = {
                "id": int(stored_move["id"]),
                "actor": "ai",
                "action": int(ai_action),
                "board_after": board_after,
            }
        except Exception as exc:
            return jsonify({"error": f"Failed to create AI opening move: {exc}"}), 500
        finally:
            _runtime().forget_game(int(game["id"]))

    return jsonify({"game": _serialize_game(game), "opening_player": opening_player, "opening_move": opening_move}), 201


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


@game_bp.get("/games/<int:game_id>/analysis")
def analyze_game(game_id: int):
    game = _repo().get_game(game_id)
    if game is None:
        return jsonify({"error": "Game not found."}), 404

    board = _decode_board(game.get("current_board_json", "[]"))
    lookahead = max(1, min(10, int(request.args.get("lookahead", 4))))
    samples = max(4, min(32, int(request.args.get("samples", 12))))

    if str(game.get("status")) != "active":
        return jsonify({"analysis": {"lookahead": lookahead, "samples": samples, "forecasts": [], "recommended_column": None}})

    forecasts = _forecast_columns(board, lookahead=lookahead, samples=samples)
    recommended_column = forecasts[0]["column"] if forecasts else None
    return jsonify(
        {
            "analysis": {
                "lookahead": lookahead,
                "samples": samples,
                "forecasts": forecasts,
                "recommended_column": recommended_column,
            }
        }
    )


@game_bp.post("/games/<int:game_id>/reset")
def reset_game(game_id: int):
    updated_game = _repo().reset_game(game_id, board=[0] * 42)
    if updated_game is None:
        return jsonify({"error": "Game not found."}), 404
    _runtime().forget_game(game_id)
    return jsonify({"game": _serialize_game(updated_game)})


@game_bp.post("/games/<int:game_id>/undo")
def undo_last_turn(game_id: int):
    game = _repo().get_game(game_id)
    if game is None:
        return jsonify({"error": "Game not found."}), 404
    try:
        result = _repo().undo_last_turn(game_id=game_id, session_index=int(game["session_index"]))
    except KeyError:
        _runtime().forget_game(game_id)
        return jsonify({"error": "Game session changed. Start a new game and try again."}), 409
    if result is None:
        return jsonify({"error": "Nothing to undo."}), 409
    updated_game, undo_info = result
    _runtime().forget_game(game_id)
    return jsonify({"game": _serialize_game(updated_game), "undo": undo_info})


@game_bp.post("/matches")
def run_match():
    """Run one non-persisted agent-vs-agent match and return a replay trace."""

    payload = request.get_json(silent=True) or {}
    agent_a_name = str(payload.get("agent_a", current_app.config["DEFAULT_AGENT"]))
    agent_b_name = str(payload.get("agent_b", _default_match_opponent(agent_a_name)))
    starting_agent = str(payload.get("starting_agent", "agent_a")).strip().lower()
    if starting_agent not in {"agent_a", "agent_b", "random"}:
        return jsonify({"error": "starting_agent must be one of: agent_a, agent_b, random."}), 400

    raw_seed = payload.get("seed")
    try:
        seed = int(raw_seed) if raw_seed is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "seed must be an integer when provided."}), 400

    raw_max_turns = payload.get("max_turns")
    try:
        max_turns = int(raw_max_turns) if raw_max_turns is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "max_turns must be an integer when provided."}), 400
    if max_turns is not None and (max_turns <= 0 or max_turns > 42):
        return jsonify({"error": "max_turns must be between 1 and 42."}), 400

    if starting_agent == "random":
        starting_agent = "agent_a" if Random(seed).random() < 0.5 else "agent_b"

    try:
        agent_a = _build_agent_from_name(agent_a_name)
        agent_b = _build_agent_from_name(agent_b_name)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400

    match = play_agent_match(
        agent_a=agent_a,
        agent_b=agent_b,
        agent_a_name=agent_a_name,
        agent_b_name=agent_b_name,
        config=_config(),
        starting_agent=starting_agent,
        max_turns=max_turns,
        seed=seed,
    )
    return jsonify({"match": match})
