from __future__ import annotations

from dataclasses import dataclass

from c4_agents.heuristic import build_heuristic_agent, list_agent_specs


@dataclass
class _Obs:
    board: list[int]
    mark: int


@dataclass
class _Cfg:
    rows: int = 6
    columns: int = 7
    inarow: int = 4
    actTimeout: float = 2.0


def test_all_registered_heuristics_return_valid_column_on_empty_board():
    obs = _Obs(board=[0] * 42, mark=1)
    cfg = _Cfg()
    for spec in list_agent_specs():
        agent = build_heuristic_agent(spec.name)
        col = int(agent(obs, cfg))
        assert 0 <= col < cfg.columns


def test_all_registered_heuristics_respect_full_columns():
    # Fill column 0 completely so chosen moves should not be 0.
    board = [0] * 42
    for row in range(6):
        board[row * 7 + 0] = 1 if row % 2 == 0 else 2
    obs = _Obs(board=board, mark=1)
    cfg = _Cfg()
    for spec in list_agent_specs():
        agent = build_heuristic_agent(spec.name)
        col = int(agent(obs, cfg))
        assert col != 0
