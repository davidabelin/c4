"""Tabular Q-learning trainer adapted from legacy ConnectX notebook workflow."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import numpy as np

try:
    import gym
    from kaggle_environments import make
except Exception as exc:  # pragma: no cover
    gym = None
    make = None
    KAGGLE_IMPORT_ERROR = str(exc)
else:
    KAGGLE_IMPORT_ERROR = None


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
    opponent: str = "negamax"
    switch_prob: float = 0.5
    seed: int = 7


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


class ConnectXEnv:
    """Thin wrapper around Kaggle ConnectX train API for notebook-parity RL."""

    def __init__(self, *, switch_prob: float, opponent: str) -> None:
        if make is None or gym is None:
            raise RuntimeError(
                "kaggle_environments/gym are required for RL training."
                + (f" Import error: {KAGGLE_IMPORT_ERROR}" if KAGGLE_IMPORT_ERROR else "")
            )
        self.env = make("connectx", debug=False)
        self.switch_prob = float(switch_prob)
        self._opponent = str(opponent)
        self._pair = [None, self._opponent]
        self._trainer = self.env.train(self._pair)
        cfg = self.env.configuration
        self.action_space = gym.spaces.Discrete(int(cfg.columns))

    def _switch(self) -> None:
        self._pair = self._pair[::-1]
        self._trainer = self.env.train(self._pair)

    def reset(self, rng: Random):
        if rng.random() < self.switch_prob:
            self._switch()
        return self._trainer.reset()

    def step(self, action: int):
        return self._trainer.step(int(action))


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
    """Train a Q-table policy against one built-in Kaggle opponent."""

    rng = Random(int(config.seed))
    np_rng = np.random.default_rng(int(config.seed))
    env = ConnectXEnv(switch_prob=config.switch_prob, opponent=config.opponent)
    q_table = QTable(env.action_space.n)

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
            valid = _valid_actions(board, env.action_space.n)
            if not valid:
                break
            if float(np_rng.random()) < epsilon:
                action = int(rng.choice(valid))
            else:
                row = q_table.row(board, mark)
                masked = [row[idx] if idx in valid else -1e12 for idx in range(env.action_space.n)]
                action = int(np.argmax(masked))

            next_state, raw_reward, done, _info = env.step(action)
            reward = _shaped_reward(done, raw_reward)

            old_value = q_table.row(board, mark)[action]
            next_max = float(np.max(q_table.row(list(next_state.board), int(next_state.mark))))
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
        "schema_version": 1,
        "model_type": "rl_qtable",
        "config": asdict(config),
        "q_table": q_table.table,
        "policy": policy,
        "metrics": {
            "episodes": int(config.episodes),
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
