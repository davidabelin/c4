"""Alpha-beta heuristic agent tuned for v9 legacy baseline behavior."""

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
    """Return one move column using depth-adaptive alpha-beta search."""

    rows = int(_cfg(config, "rows", 6))
    columns = int(_cfg(config, "columns", 7))
    inarow = int(_cfg(config, "inarow", 4))

    # Heuristic weights.
    w_self_2 = 2
    w_self_3 = 20
    w_self_4 = 200
    w_opp_2 = -1
    w_opp_3 = -10
    w_opp_4 = -100

    if list(obs.board).count(0) >= rows * columns // 2:
        n_steps = 2
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

    def count_windows(grid: np.ndarray, num_discs: int, piece: int) -> int:
        count = 0
        # Horizontal
        for row in range(rows):
            for col in range(columns - (inarow - 1)):
                window = list(grid[row, col : col + inarow])
                if check_window(window, num_discs, piece):
                    count += 1
        # Vertical
        for row in range(rows - (inarow - 1)):
            for col in range(columns):
                window = list(grid[row : row + inarow, col])
                if check_window(window, num_discs, piece):
                    count += 1
        # Positive diagonal
        for row in range(rows - (inarow - 1)):
            for col in range(columns - (inarow - 1)):
                window = list(grid[range(row, row + inarow), range(col, col + inarow)])
                if check_window(window, num_discs, piece):
                    count += 1
        # Negative diagonal
        for row in range(inarow - 1, rows):
            for col in range(columns - (inarow - 1)):
                window = list(grid[range(row, row - inarow, -1), range(col, col + inarow)])
                if check_window(window, num_discs, piece):
                    count += 1
        return count

    def get_score(grid: np.ndarray, mark: int) -> tuple[float, bool]:
        self_twos = count_windows(grid, 2, mark)
        self_threes = count_windows(grid, 3, mark)
        self_fours = count_windows(grid, 4, mark)
        opp_twos = count_windows(grid, 2, mark % 2 + 1)
        opp_threes = count_windows(grid, 3, mark % 2 + 1)
        opp_fours = count_windows(grid, 4, mark % 2 + 1)
        score = (
            w_self_2 * self_twos
            + w_self_3 * self_threes
            + w_self_4 * self_fours
            + w_opp_2 * opp_twos
            + w_opp_3 * opp_threes
            + w_opp_4 * opp_fours
        )
        is_terminal = (self_fours != 0) or (opp_fours != 0) or (list(grid[0, :]).count(0) == 0)
        return float(score), bool(is_terminal)

    def alphabeta(node: np.ndarray, depth: int, alpha: float, beta: float, maximizing: bool, mark: int) -> float:
        node_score, is_terminal = get_score(node, mark)
        if depth == 0 or is_terminal:
            return float(node_score)
        valid_moves = [c for c in range(columns) if int(node[0][c]) == 0]
        if maximizing:
            value = -np.inf
            for col in valid_moves:
                child = drop_piece(node, col, mark)
                value = max(value, alphabeta(child, depth - 1, alpha, beta, False, mark))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return float(value)
        value = np.inf
        for col in valid_moves:
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

    # Quick terminal check: if we can win immediately, take it.
    for col in valid_moves:
        _, terminal = get_score(drop_piece(board, col, int(obs.mark)), int(obs.mark))
        if terminal:
            if debug:
                print(f"Immediate terminal move detected at column {col}")
            return int(col)

    scores = {col: score_move(board, col, int(obs.mark), n_steps) for col in valid_moves}
    best_score = max(scores.values())
    max_cols = [col for col, score in scores.items() if score == best_score]
    for preferred in (3, 4, 2):
        if preferred in max_cols:
            return int(preferred)
    return int(random.choice(max_cols))


agent = my_agent

__all__ = ["my_agent", "agent"]
