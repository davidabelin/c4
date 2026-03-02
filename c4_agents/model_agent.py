"""Model-backed agent wrapper used for ``active_model`` gameplay."""

from __future__ import annotations

from random import Random

from c4_core.board import valid_columns
from c4_core.types import Connect4Config
from c4_rl.trainer import state_key
from c4_training.supervised import load_artifact, predict_action


class ModelBackedAgent:
    """Serve a trained artifact through a Kaggle-compatible callable signature."""

    name = "active_model"

    def __init__(self, artifact_path: str) -> None:
        self._artifact_path = artifact_path
        self._artifact = load_artifact(artifact_path)
        self._rng = Random()
        self._history_actions: list[int] = []

    def reset(self, seed: int | None = None) -> None:
        self._rng.seed(seed)
        self._history_actions = []

    def _fallback_action(self, board: list[int], config: Connect4Config) -> int:
        valid = valid_columns(board, config)
        for preferred in (3, 4, 2, 5, 1, 6, 0):
            if preferred in valid:
                return preferred
        return self._rng.choice(valid) if valid else 0

    def __call__(self, obs, config) -> int:
        board = list(obs.board)
        mark = int(getattr(obs, "mark", 2))
        cfg = config if isinstance(config, Connect4Config) else Connect4Config(
            rows=int(getattr(config, "rows", 6) if not isinstance(config, dict) else config.get("rows", 6)),
            columns=int(getattr(config, "columns", 7) if not isinstance(config, dict) else config.get("columns", 7)),
            inarow=int(getattr(config, "inarow", 4) if not isinstance(config, dict) else config.get("inarow", 4)),
        )
        valid = valid_columns(board, cfg)
        if not valid:
            return 0

        model_type = str(self._artifact.get("model_type", "decision_tree"))
        if model_type == "rl_qtable":
            policy = self._artifact.get("policy") or {}
            key = state_key(board, mark)
            action = policy.get(key)
            if action is not None and int(action) in valid:
                chosen = int(action)
            else:
                q_table = self._artifact.get("q_table") or {}
                row = q_table.get(key)
                if isinstance(row, list) and row:
                    ranked = sorted(range(len(row)), key=lambda idx: float(row[idx]), reverse=True)
                    chosen = next((idx for idx in ranked if idx in valid), self._fallback_action(board, cfg))
                else:
                    chosen = self._fallback_action(board, cfg)
        else:
            predicted = predict_action(
                self._artifact,
                board=board,
                history_actions=self._history_actions,
                valid_moves=valid,
            )
            chosen = int(predicted) if predicted is not None else self._fallback_action(board, cfg)

        self._history_actions.append(int(chosen))
        if len(self._history_actions) > 32:
            self._history_actions = self._history_actions[-32:]
        return int(chosen)
