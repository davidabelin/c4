"""Tabular Q-learning trainer using the local Connect4 engine."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import numpy as np

from c4_core.board import board_to_grid, drop_piece, has_any_four, valid_columns
from c4_core.engine import select_ai_action
from c4_core.types import Connect4Config


@dataclass(slots=True)
class QTrainConfig:
    """Configuration for tabular Q-learning training against a fixed opponent."""

    episodes: int = 5000
    alpha: float = 0.1
    gamma: float = 0.6
    epsilon_start: float = 0.99
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.9999
    alpha_decay_step: int = 1000
    alpha_decay_rate: float = 0.9
    opponent: str = "alpha_beta_v9"
    switch_prob: float = 0.5
    seed: int = 7


@dataclass(slots=True)
class ConnectXState:
    board: list[int]
    mark: int


def state_key(board: list[int], mark: int) -> str:
    """Encode board+mark state into compact base-3 hash key."""

    tokens = board[:] + [int(mark)]
    as_base3 = "".join(str(int(value)) for value in tokens)
    return hex(int(as_base3, 3))[2:]


class QTable:
    """Sparse dictionary-backed Q-table keyed by board+mark state hash."""

    def __init__(self, action_space_n: int) -> None:
        self.action_space_n = int(action_space_n)
        self.table: dict[str, list[float]] = {}

    def row(self, board: list[int], mark: int) -> list[float]:
        key = state_key(board, mark)
        if key not in self.table:
            self.table[key] = list(np.zeros(self.action_space_n))
        return self.table[key]


def _random_agent(obs, config) -> int:
    valid = valid_columns(list(obs.board), config)
    if not valid:
        return 0
    for preferred in (3, 4, 2, 5, 1, 6, 0):
        if preferred in valid:
            return preferred
    return int(valid[0])


def _resolve_opponent_agent(opponent_name: str):
    normalized = str(opponent_name or "alpha_beta_v9").strip().lower()
    if normalized == "random":
        return "random", _random_agent

    alias = {"negamax": "alpha_beta_v9"}.get(normalized, normalized)
    from c4_agents.heuristic import AGENT_SPECS, build_heuristic_agent

    if alias not in AGENT_SPECS:
        raise ValueError(f"Unknown RL opponent '{opponent_name}'.")
    return alias, build_heuristic_agent(alias)


class ConnectXEnv:
    """Local Connect4 environment for learner-vs-opponent episodes."""

    def __init__(self, *, switch_prob: float, opponent: str) -> None:
        self.config = Connect4Config(rows=6, columns=7, inarow=4)
        self.switch_prob = float(switch_prob)
        self.action_space_n = int(self.config.columns)
        self.opponent_name, self.opponent_agent = _resolve_opponent_agent(opponent)
        self._board = [0] * (int(self.config.rows) * int(self.config.columns))
        self._learner_mark = 1
        self._opponent_mark = 2
        self._rng = Random()

    def _reset_opponent(self, seed: int | None) -> None:
        if hasattr(self.opponent_agent, "reset"):
            try:
                self.opponent_agent.reset(seed=seed)
            except TypeError:
                self.opponent_agent.reset(seed)

    def reset(self, rng: Random) -> ConnectXState:
        self._board = [0] * (int(self.config.rows) * int(self.config.columns))
        episode_seed = int(rng.randrange(0, 2**31))
        self._rng = Random(episode_seed)
        learner_starts = self._rng.random() >= self.switch_prob
        self._learner_mark = 1 if learner_starts else 2
        self._opponent_mark = 2 if learner_starts else 1
        self._reset_opponent(episode_seed)

        if not learner_starts:
            opening = select_ai_action(
                self.opponent_agent,
                self._board,
                config=self.config,
                mark=self._opponent_mark,
                rng=self._rng,
            )
            grid = drop_piece(board_to_grid(self._board, self.config), opening, mark=self._opponent_mark, config=self.config)
            self._board = grid.reshape(-1).astype(int).tolist()

        return ConnectXState(board=list(self._board), mark=int(self._learner_mark))

    def step(self, action: int) -> tuple[ConnectXState, float, bool, dict]:
        valid = valid_columns(self._board, self.config)
        if int(action) not in valid:
            return ConnectXState(board=list(self._board), mark=int(self._learner_mark)), -1.0, True, {"illegal_move": True}

        learner_grid = drop_piece(
            board_to_grid(self._board, self.config),
            int(action),
            mark=self._learner_mark,
            config=self.config,
        )
        learner_board = learner_grid.reshape(-1).astype(int).tolist()
        if has_any_four(learner_grid, mark=self._learner_mark, config=self.config):
            self._board = learner_board
            return ConnectXState(board=list(self._board), mark=int(self._learner_mark)), 1.0, True, {"winner": "learner"}
        if not valid_columns(learner_board, self.config):
            self._board = learner_board
            return ConnectXState(board=list(self._board), mark=int(self._learner_mark)), 0.0, True, {"winner": "tie"}

        opponent_action = select_ai_action(
            self.opponent_agent,
            learner_board,
            config=self.config,
            mark=self._opponent_mark,
            rng=self._rng,
        )
        opponent_grid = drop_piece(
            board_to_grid(learner_board, self.config),
            opponent_action,
            mark=self._opponent_mark,
            config=self.config,
        )
        self._board = opponent_grid.reshape(-1).astype(int).tolist()
        if has_any_four(opponent_grid, mark=self._opponent_mark, config=self.config):
            return ConnectXState(board=list(self._board), mark=int(self._learner_mark)), -1.0, True, {"winner": "opponent"}
        if not valid_columns(self._board, self.config):
            return ConnectXState(board=list(self._board), mark=int(self._learner_mark)), 0.0, True, {"winner": "tie"}

        return ConnectXState(board=list(self._board), mark=int(self._learner_mark)), 0.0, False, {}


def _valid_actions(board: list[int], columns: int) -> list[int]:
    return [col for col in range(int(columns)) if int(board[col]) == 0]


def _shaped_reward(done: bool, reward: float | int | None) -> float:
    if done:
        if reward is None:
            return -20.0
        reward_value = float(reward)
        if reward_value > 0:
            return 20.0
        if reward_value < 0:
            return -20.0
        return 10.0
    return -0.05


def train_q_table(config: QTrainConfig) -> dict:
    """Train a Q-table policy against a local opponent policy."""

    rng = Random(int(config.seed))
    np_rng = np.random.default_rng(int(config.seed))
    env = ConnectXEnv(switch_prob=config.switch_prob, opponent=config.opponent)
    q_table = QTable(env.action_space_n)

    epsilon = float(config.epsilon_start)
    alpha = float(config.alpha)
    episode_lengths: list[int] = []
    total_rewards: list[float] = []
    rolling_rewards: list[float] = []

    for episode in range(int(config.episodes)):
        state = env.reset(rng)
        done = False
        steps = 0
        cumulative = 0.0
        while not done:
            board = list(state.board)
            mark = int(state.mark)
            valid = _valid_actions(board, env.action_space_n)
            if not valid:
                break
            if float(np_rng.random()) < epsilon:
                action = int(rng.choice(valid))
            else:
                row = q_table.row(board, mark)
                masked = [row[idx] if idx in valid else -1e12 for idx in range(env.action_space_n)]
                action = int(np.argmax(masked))

            next_state, raw_reward, done, _info = env.step(action)
            reward = _shaped_reward(done, raw_reward)

            old_value = q_table.row(board, mark)[action]
            next_max = 0.0 if done else float(np.max(q_table.row(list(next_state.board), int(next_state.mark))))
            new_value = (1.0 - alpha) * old_value + alpha * (reward + float(config.gamma) * next_max)
            q_table.row(board, mark)[action] = float(new_value)

            state = next_state
            cumulative += reward
            steps += 1

        epsilon = max(float(config.epsilon_end), epsilon * float(config.epsilon_decay))
        if (episode + 1) % int(config.alpha_decay_step) == 0:
            alpha *= float(config.alpha_decay_rate)

        episode_lengths.append(steps)
        total_rewards.append(cumulative)
        window = total_rewards[max(0, len(total_rewards) - 100) :]
        rolling_rewards.append(float(sum(window) / max(1, len(window))))

    policy = {key: int(np.argmax(values)) for key, values in q_table.table.items()}
    return {
        "schema_version": 2,
        "model_type": "rl_qtable",
        "config": asdict(config),
        "q_table": q_table.table,
        "policy": policy,
        "metrics": {
            "episodes": int(config.episodes),
            "opponent": env.opponent_name,
            "q_table_states": len(q_table.table),
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "mean_total_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
            "final_100_avg_reward": float(rolling_rewards[-1]) if rolling_rewards else 0.0,
            "final_alpha": float(alpha),
            "final_epsilon": float(epsilon),
        },
    }


def save_artifact(path: str | Path, artifact: dict) -> Path:
    """Serialize RL artifact as pickle file."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(artifact))
    return out


def load_artifact(path: str | Path) -> dict:
    """Load serialized RL artifact from pickle file."""

    return pickle.loads(Path(path).read_bytes())
