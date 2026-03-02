"""Board helpers shared by agents/training for Connect4."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from c4_core.types import Connect4Config


def board_to_grid(board: Iterable[int], config: Connect4Config) -> np.ndarray:
    """Convert flat board list into 2D numpy grid."""

    arr = np.asarray(list(board), dtype=int)
    expected = int(config.rows) * int(config.columns)
    if arr.size != expected:
        raise ValueError(f"Board size must be {expected}, got {arr.size}")
    return arr.reshape(int(config.rows), int(config.columns))


def valid_columns(board: Iterable[int], config: Connect4Config) -> list[int]:
    """Return legal columns from the current board."""

    flat = list(board)
    return [col for col in range(int(config.columns)) if flat[col] == 0]


def drop_piece(grid: np.ndarray, col: int, mark: int, config: Connect4Config) -> np.ndarray:
    """Return next board grid after dropping one piece in a column."""

    next_grid = np.array(grid, copy=True)
    row_index = None
    for row in range(int(config.rows) - 1, -1, -1):
        if int(next_grid[row, col]) == 0:
            row_index = row
            break
    if row_index is None:
        raise ValueError(f"Column {col} is full")
    next_grid[row_index, col] = int(mark)
    return next_grid


def has_any_four(grid: np.ndarray, mark: int, config: Connect4Config) -> bool:
    """Return True when `mark` has an in-a-row line on board."""

    rows = int(config.rows)
    cols = int(config.columns)
    k = int(config.inarow)

    # Horizontal
    for row in range(rows):
        for col in range(cols - (k - 1)):
            if np.all(grid[row, col : col + k] == mark):
                return True
    # Vertical
    for row in range(rows - (k - 1)):
        for col in range(cols):
            if np.all(grid[row : row + k, col] == mark):
                return True
    # Positive slope diagonal
    for row in range(rows - (k - 1)):
        for col in range(cols - (k - 1)):
            if all(int(grid[row + i, col + i]) == int(mark) for i in range(k)):
                return True
    # Negative slope diagonal
    for row in range(k - 1, rows):
        for col in range(cols - (k - 1)):
            if all(int(grid[row - i, col + i]) == int(mark) for i in range(k)):
                return True
    return False
