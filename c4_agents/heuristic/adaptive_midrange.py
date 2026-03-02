"""Adaptive midrange heuristic agent with depth scaling."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def _cfg(config: Any, key: str, default: int | float | None = None):
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def my_agent(obs, config, n_steps: int = 2, debug: bool = False) -> int:
    """Return one move column using adaptive weighted alpha-beta search."""

    rows = int(_cfg(config, "rows", 6))
    columns = int(_cfg(config, "columns", 7))
    inarow = int(_cfg(config, "inarow", 4))

    # Heuristic weights.
    w_self_2 = 2
    w_self_3 = 20
    w_self_4 = 2000
    w_opp_2 = -1
    w_opp_3 = -10
    w_opp_4 = -100
    w_mid_2 = 10
    w_mid_2_opp = -10

    if list(obs.board).count(0) >= 6 * (rows * columns // 7):
        n_steps = 2
    else:
        n_steps = 3

    top_row = list(obs.board[0:columns])
    if top_row.count(0) >= 6:
        n_steps = 3
    elif top_row.count(0) >= 5:
        n_steps = 4
    elif top_row.count(0) >= 4:
        n_steps = 5
    elif top_row.count(0) >= 3:
        n_steps = 6
    else:
        n_steps = 3

    def drop_piece(grid: np.ndarray, col: int, mark: int) -> np.ndarray:
        next_grid = np.array(grid, copy=True)
        for row in range(rows - 1, -1, -1):
            if int(next_grid[row][col]) == 0:
                next_grid[row][col] = int(mark)
                return next_grid
        raise ValueError(f"Column {col} is full")

    def check_window(window: list[int], num_discs: int, piece: int) -> bool:
        return window.count(piece) == num_discs and window.count(0) == inarow - num_discs

    def count_windows(grid: np.ndarray, num_discs: int, piece: int, check_mid: bool = False) -> int:
        count = 0
        # Horizontal
        for row in range(rows):
            for col in range(columns - (inarow - 1)):
                window = list(grid[row, col : col + inarow])
                if check_window(window, num_discs, piece):
                    if not check_mid or (window[0] == 0 and window[-1] == 0):
                        count += 1
        # Vertical
        for row in range(rows - (inarow - 1)):
            for col in range(columns):
                window = list(grid[row : row + inarow, col])
                if check_window(window, num_discs, piece):
                    if not check_mid or (window[0] == 0 and window[-1] == 0):
                        count += 1
        # Positive diagonal
        for row in range(rows - (inarow - 1)):
            for col in range(columns - (inarow - 1)):
                window = list(grid[range(row, row + inarow), range(col, col + inarow)])
                if check_window(window, num_discs, piece):
                    if not check_mid or (window[0] == 0 and window[-1] == 0):
                        count += 1
        # Negative diagonal
        for row in range(inarow - 1, rows):
            for col in range(columns - (inarow - 1)):
                window = list(grid[range(row, row - inarow, -1), range(col, col + inarow)])
                if check_window(window, num_discs, piece):
                    if not check_mid or (window[0] == 0 and window[-1] == 0):
                        count += 1
        return count

    def check_terminal(grid: np.ndarray, mark: int) -> bool:
        return count_windows(grid, 4, mark) != 0 or list(grid[0, :]).count(0) == 0

    def get_score(grid: np.ndarray, mark: int) -> tuple[float, bool]:
        if list(grid[0, :]).count(0) == 0:
            return 0.0, True
        self_fours = count_windows(grid, 4, mark)
        opp_fours = count_windows(grid, 4, mark % 2 + 1)
        if self_fours != 0 or opp_fours != 0:
            return float(w_self_4 * self_fours + w_opp_4 * opp_fours), True
        self_threes = count_windows(grid, 3, mark)
        opp_threes = count_windows(grid, 3, mark % 2 + 1)
        self_twos = count_windows(grid, 2, mark)
        opp_twos = count_windows(grid, 2, mark % 2 + 1)
        mid_twos = count_windows(grid, 2, mark, check_mid=True)
        mid_twos_opp = count_windows(grid, 2, mark % 2 + 1, check_mid=True)
        score = (
            w_self_2 * self_twos
            + w_self_3 * self_threes
            + w_opp_2 * opp_twos
            + w_opp_3 * opp_threes
            + w_mid_2 * mid_twos
            + w_mid_2_opp * mid_twos_opp
        )
        return float(score), False

    def alphabeta(node: np.ndarray, depth: int, alpha: float, beta: float, maximizing: bool, mark: int) -> float:
        node_score, is_terminal = get_score(node, mark)
        if depth == 0 or is_terminal:
            return float(node_score)
        valid = [c for c in range(columns) if int(node[0][c]) == 0]
        if maximizing:
            value = -np.inf
            for col in valid:
                child = drop_piece(node, col, mark)
                value = max(value, alphabeta(child, depth - 1, alpha, beta, False, mark))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return float(value)
        value = np.inf
        for col in valid:
            child = drop_piece(node, col, mark % 2 + 1)
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True, mark))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return float(value)

    def score_move(grid: np.ndarray, col: int, mark: int, depth: int) -> float:
        return float(alphabeta(drop_piece(grid, col, mark), depth - 1, -np.inf, np.inf, False, mark))

    board = np.asarray(obs.board).reshape(rows, columns)
    valid_moves = [c for c in range(columns) if int(obs.board[c]) == 0]
    if not valid_moves:
        return 0

    # Quick pass: immediate win or immediate block.
    for col in valid_moves:
        if check_terminal(drop_piece(board, col, int(obs.mark)), int(obs.mark)):
            return int(col)
        if check_terminal(drop_piece(board, col, int(obs.mark) % 2 + 1), int(obs.mark) % 2 + 1):
            return int(col)

    scores = {col: score_move(board, col, int(obs.mark), n_steps) for col in valid_moves}
    max_score = max(scores.values())
    max_cols = [col for col, score in scores.items() if score == max_score]
    for preferred in (3, 4, 2):
        if preferred in max_cols:
            return int(preferred)
    if debug:
        print(f"scores={scores}")
    return int(random.choice(max_cols))


agent = my_agent

__all__ = ["my_agent", "agent"]
