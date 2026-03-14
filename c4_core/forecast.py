"""Heuristic Connect4 column forecast helpers shared by play and arena views."""

from __future__ import annotations

import math
import random as pyrandom
from random import Random

from c4_agents import build_heuristic_agent
from c4_core.board import board_to_grid, drop_piece, has_any_four, valid_columns
from c4_core.engine import select_ai_action
from c4_core.types import Connect4Config


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


def terminal_outcome_for_board(board: list[int], *, config: Connect4Config) -> str | None:
    grid = board_to_grid(board, config)
    if has_any_four(grid, mark=1, config=config):
        return "player"
    if has_any_four(grid, mark=2, config=config):
        return "ai"
    if not valid_columns(board, config):
        return "tie"
    return None


def _heuristic_probability(board: list[int], *, perspective_mark: int, config: Connect4Config) -> float:
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


def _outcome_to_probability(outcome: str, *, perspective_mark: int) -> float:
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
    config: Connect4Config,
) -> float:
    working = list(board)
    current_mark = int(next_mark)
    seed_value = 1009 + lookahead * 17 + sum((idx + 1) * int(value) for idx, value in enumerate(board))
    saved_state = pyrandom.getstate()
    pyrandom.seed(seed_value)
    try:
        agent = build_heuristic_agent("alpha_beta_v9")
        for ply in range(max(0, int(lookahead))):
            terminal = terminal_outcome_for_board(working, config=config)
            if terminal is not None:
                return _outcome_to_probability(terminal, perspective_mark=perspective_mark)
            action = select_ai_action(agent, working, config=config, mark=current_mark, rng=Random(seed_value + ply))
            next_grid = drop_piece(board_to_grid(working, config), action, mark=current_mark, config=config)
            working = next_grid.reshape(-1).astype(int).tolist()
            current_mark = 1 if current_mark == 2 else 2
    finally:
        pyrandom.setstate(saved_state)

    terminal = terminal_outcome_for_board(working, config=config)
    if terminal is not None:
        return _outcome_to_probability(terminal, perspective_mark=perspective_mark)
    return _heuristic_probability(working, perspective_mark=perspective_mark, config=config)


def forecast_columns(
    board: list[int],
    *,
    perspective_mark: int,
    lookahead: int,
    config: Connect4Config,
) -> list[dict]:
    """Return one independent win estimate for each legal column."""

    forecasts: list[dict] = []
    opponent_mark = 1 if int(perspective_mark) == 2 else 2
    for column in valid_columns(board, config):
        next_grid = drop_piece(board_to_grid(board, config), column, mark=int(perspective_mark), config=config)
        board_after_choice = next_grid.reshape(-1).astype(int).tolist()
        immediate = terminal_outcome_for_board(board_after_choice, config=config)
        if immediate is not None:
            win_estimate = _outcome_to_probability(immediate, perspective_mark=int(perspective_mark))
        else:
            win_estimate = _simulate_guided_outcome(
                board_after_choice,
                next_mark=opponent_mark,
                perspective_mark=int(perspective_mark),
                lookahead=max(0, int(lookahead) - 1),
                config=config,
            )
        forecasts.append(
            {
                "column": int(column),
                "win_estimate": round(float(win_estimate), 4),
                "label": f"{round(float(win_estimate) * 100)}%",
            }
        )
    forecasts.sort(key=lambda item: (-float(item["win_estimate"]), int(item["column"])))
    return forecasts
