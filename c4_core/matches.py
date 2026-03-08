"""Agent-vs-agent Connect4 match helpers for replayable sessions."""

from __future__ import annotations

from random import Random
from typing import Any, Callable

from c4_core.board import board_to_grid, drop_piece, has_any_four, valid_columns
from c4_core.engine import select_ai_action
from c4_core.types import Connect4Config


def _reset_agent(agent: Any, seed: int | None) -> None:
    if hasattr(agent, "reset"):
        try:
            agent.reset(seed=seed)
        except TypeError:
            agent.reset(seed)


def _observe_agent(agent: Any, payload: dict) -> None:
    if not hasattr(agent, "observe"):
        return
    try:
        agent.observe(payload)
    except Exception:
        return


def play_agent_match(
    *,
    agent_a: Any,
    agent_b: Any,
    agent_a_name: str,
    agent_b_name: str,
    config: Connect4Config,
    starting_agent: str = "agent_a",
    max_turns: int | None = None,
    seed: int | None = None,
    on_move: Callable[[dict], None] | None = None,
) -> dict:
    """Run one replayable Connect4 game between two agents."""

    total_cells = int(config.rows) * int(config.columns)
    turn_limit = total_cells if max_turns is None else int(max_turns)
    if turn_limit <= 0:
        raise ValueError("max_turns must be a positive integer.")

    seed_rng = Random(seed)
    _reset_agent(agent_a, seed_rng.randrange(0, 2**31) if seed is not None else None)
    _reset_agent(agent_b, seed_rng.randrange(0, 2**31) if seed is not None else None)

    if starting_agent not in {"agent_a", "agent_b"}:
        raise ValueError("starting_agent must be either 'agent_a' or 'agent_b'.")

    marks = {"agent_a": 1 if starting_agent == "agent_a" else 2, "agent_b": 1 if starting_agent == "agent_b" else 2}
    board = [0] * total_cells
    active = starting_agent
    trace: list[dict] = []
    winner: str | None = None
    status = "completed"

    for move_index in range(min(turn_limit, total_cells)):
        actor = active
        agent = agent_a if actor == "agent_a" else agent_b
        mark = int(marks[actor])
        board_before = list(board)
        action = select_ai_action(agent, board_before, config=config, mark=mark, rng=seed_rng)
        next_grid = drop_piece(board_to_grid(board_before, config), action, mark=mark, config=config)
        board_after = next_grid.reshape(-1).astype(int).tolist()
        remaining = valid_columns(board_after, config)

        if has_any_four(next_grid, mark=mark, config=config):
            outcome = actor
            winner = actor
        elif not remaining:
            outcome = "tie"
            winner = "tie"
        else:
            outcome = "ongoing"

        frame = {
            "move_index": move_index,
            "actor": actor,
            "mark": mark,
            "action": int(action),
            "board_before": board_before,
            "board_after": board_after,
            "outcome": outcome,
        }
        trace.append(frame)
        if on_move is not None:
            on_move(dict(frame))

        _observe_agent(
            agent,
            {
                "actor": actor,
                "mark": mark,
                "action": int(action),
                "board_before": board_before,
                "board_after": board_after,
                "outcome": outcome,
                "config": config,
            },
        )

        board = board_after
        if outcome != "ongoing":
            break
        active = "agent_b" if active == "agent_a" else "agent_a"
    else:
        status = "truncated"

    if winner is None and status == "completed":
        winner = "tie"

    return {
        "mode": "agent_vs_agent",
        "agent_a": agent_a_name,
        "agent_b": agent_b_name,
        "starting_agent": starting_agent,
        "mark_agent_a": int(marks["agent_a"]),
        "mark_agent_b": int(marks["agent_b"]),
        "seed": seed,
        "max_turns": turn_limit,
        "moves_played": len(trace),
        "status": status,
        "winner": winner,
        "final_board": board,
        "trace": trace,
    }
