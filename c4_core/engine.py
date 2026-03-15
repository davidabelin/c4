"""Gameplay execution helpers for human-vs-agent Connect4 sessions.

Role
----
This module is the pure gameplay layer used by the Connect4 web/API surface.
It validates moves, applies board transitions, invokes the selected AI agent,
and emits structured turn results without performing any persistence.

Cross-Repo Context
------------------
`c4_core.engine` fills the same role for Connect4 that `rps_core.engine` fills
for RPS: deterministic gameplay resolution under a storage-agnostic contract
that the web layer can persist and replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from types import SimpleNamespace
from typing import Any

import numpy as np

from c4_core.board import board_to_grid, drop_piece, has_any_four, valid_columns
from c4_core.types import Connect4Config, normalize_column


@dataclass(slots=True)
class TurnResult:
    """One resolved human turn, including an optional AI reply.

    Notes
    -----
    The Connect4 UI thinks in player turns, but storage records individual
    moves. This dataclass bridges those views by carrying both the grouped turn
    outcome and the intermediate board snapshots needed for persistence.
    """

    round_index: int
    player_action: int
    ai_action: int | None
    outcome: str
    reward_delta: int
    board_before_player: list[int]
    board_after_player: list[int]
    board_after_ai: list[int] | None


def _to_obs(board: list[int], mark: int):
    """Wrap a raw board list in the minimal observation shape agents expect."""

    return SimpleNamespace(board=list(board), mark=int(mark))


def _legal_fallback(valid: list[int], *, rng: Random) -> int:
    """Choose a safe fallback action when an agent errors or proposes junk.

    Notes
    -----
    The fallback is center-biased before it becomes random so degraded play
    still looks plausible and avoids obvious edge-column drift.
    """

    # Prefer center, then near-center, then random.
    for preferred in (3, 4, 2, 5, 1, 6, 0):
        if preferred in valid:
            return int(preferred)
    return int(rng.choice(valid))


def select_ai_action(
    agent: Any,
    board: list[int],
    *,
    config: Connect4Config,
    mark: int = 2,
    rng: Random | None = None,
) -> int:
    """Select one legal AI move from an agent callable and board state.

    Role
    ----
    This function isolates all "call the agent, catch bad outputs, clamp to a
    legal move" logic so the rest of the turn pipeline can assume it receives a
    valid column index.
    """

    random = rng or Random()
    valid = valid_columns(board, config)
    if not valid:
        raise ValueError("No valid moves are available.")
    obs = _to_obs(board, mark)
    try:
        raw = int(agent(obs, config))
    except Exception:
        raw = _legal_fallback(valid, rng=random)
    if raw not in valid:
        raw = _legal_fallback(valid, rng=random)
    return int(raw)


def play_human_turn(
    *,
    board: list[int],
    player_action: int,
    agent: Any,
    config: Connect4Config,
    round_index: int,
    rng: Random | None = None,
) -> TurnResult:
    """Resolve one full player turn: player move, then optional AI move.

    Role
    ----
    This is the main low-level gameplay path used by the Connect4 play API. It
    emits enough board snapshots for the repository layer to persist each move
    row without recomputing transitions.
    """

    random = rng or Random()
    if len(board) != int(config.rows) * int(config.columns):
        raise ValueError("Invalid board size.")
    valid = valid_columns(board, config)
    if not valid:
        raise ValueError("Game is already full; reset before playing another move.")
    player_col = normalize_column(player_action, config)
    if player_col not in valid:
        raise ValueError(f"Column {player_col} is not currently playable.")

    grid_before = board_to_grid(board, config)
    grid_after_player = drop_piece(grid_before, player_col, mark=1, config=config)
    board_after_player = grid_after_player.reshape(-1).astype(int).tolist()

    if has_any_four(grid_after_player, mark=1, config=config):
        return TurnResult(
            round_index=int(round_index),
            player_action=int(player_col),
            ai_action=None,
            outcome="player",
            reward_delta=1,
            board_before_player=list(board),
            board_after_player=board_after_player,
            board_after_ai=None,
        )
    if not valid_columns(board_after_player, config):
        return TurnResult(
            round_index=int(round_index),
            player_action=int(player_col),
            ai_action=None,
            outcome="tie",
            reward_delta=0,
            board_before_player=list(board),
            board_after_player=board_after_player,
            board_after_ai=None,
        )

    ai_col = select_ai_action(agent, board_after_player, config=config, mark=2, rng=random)
    grid_after_ai = drop_piece(grid_after_player, ai_col, mark=2, config=config)
    board_after_ai = grid_after_ai.reshape(-1).astype(int).tolist()

    if has_any_four(grid_after_ai, mark=2, config=config):
        outcome = "ai"
        reward = -1
    elif not valid_columns(board_after_ai, config):
        outcome = "tie"
        reward = 0
    else:
        outcome = "ongoing"
        reward = 0

    return TurnResult(
        round_index=int(round_index),
        player_action=int(player_col),
        ai_action=int(ai_col),
        outcome=outcome,
        reward_delta=int(reward),
        board_before_player=list(board),
        board_after_player=board_after_player,
        board_after_ai=board_after_ai,
    )


def replay_ai_agent_state(agent: Any, moves: list[dict], config: Connect4Config) -> None:
    """Replay historic AI moves into stateful agents when supported.

    Most current heuristic agents are stateless callables, so this function is
    intentionally lightweight. It only calls optional `reset`/`observe`
    methods.

    Cross-Repo Context
    ------------------
    This is the Connect4 analogue of replaying RPS rounds into an agent before
    acting. The difference is that many Connect4 agents are stateless board
    evaluators, so this helper degrades cleanly when no stateful hooks exist.
    """

    if hasattr(agent, "reset"):
        try:
            agent.reset(seed=None)
        except Exception:
            pass
    if not hasattr(agent, "observe"):
        return
    for row in moves:
        if str(row.get("actor", "")) != "ai":
            continue
        try:
            board_before = row.get("board_before_json")
            board_after = row.get("board_after_json")
            if isinstance(board_before, str):
                import json

                board_before = json.loads(board_before)
            if isinstance(board_after, str):
                import json

                board_after = json.loads(board_after)
            payload = {
                "action": int(row["action"]),
                "board_before": list(board_before) if board_before is not None else None,
                "board_after": list(board_after) if board_after is not None else None,
                "outcome": row.get("outcome"),
                "config": config,
            }
            agent.observe(payload)
        except Exception:
            continue
