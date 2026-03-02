"""Typed primitives for Connect4 game state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Connect4Config:
    """Static board/game configuration."""

    rows: int = 6
    columns: int = 7
    inarow: int = 4


@dataclass(slots=True)
class Connect4Observation:
    """Observation payload compatible with Kaggle-style agent signatures."""

    board: list[int]
    mark: int


def normalize_column(value: int | str, config: Connect4Config) -> int:
    """Validate and normalize a requested move column index."""

    column = int(value)
    if column < 0 or column >= int(config.columns):
        raise ValueError(f"Column must be in [0, {int(config.columns) - 1}], got {column}")
    return column
