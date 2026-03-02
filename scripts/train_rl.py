"""CLI entry point for c4 tabular RL policy training."""

from __future__ import annotations

import argparse
import pickle
from datetime import UTC, datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from c4_rl.trainer import QTrainConfig, train_q_table
from c4_storage import C4Repository
from c4_storage.object_store import is_gcs_uri, join_storage_path, write_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tabular RL policy (Q-learning) for c4.")
    parser.add_argument("--db-path", type=str, default="data/c4.db")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--epsilon-start", type=float, default=0.99)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--alpha-decay-step", type=int, default=1000)
    parser.add_argument("--alpha-decay-rate", type=float, default=0.9)
    parser.add_argument("--opponent", type=str, default="negamax")
    parser.add_argument("--switch-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    repository = C4Repository(args.db_path)
    repository.init_schema()
    config = QTrainConfig(
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        alpha_decay_step=args.alpha_decay_step,
        alpha_decay_rate=args.alpha_decay_rate,
        opponent=args.opponent,
        switch_prob=args.switch_prob,
        seed=args.seed,
    )

    models_dir = str(args.models_dir)
    if not is_gcs_uri(models_dir):
        Path(models_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = join_storage_path(models_dir, f"rl_qtable_cli_{timestamp}.pkl")
    artifact = train_q_table(config=config)
    write_bytes(artifact_path, pickle.dumps(artifact), content_type="application/octet-stream")
    metrics = dict(artifact.get("metrics", {}))
    metrics["artifact_path"] = artifact_path
    model_row = repository.create_model(
        name=f"rl-qtable-cli-{timestamp}",
        model_type="rl_qtable",
        artifact_path=artifact_path,
        lookback=1,
        metrics=metrics,
    )
    print("RL training completed")
    print(f"Model ID: {model_row['id']}")
    print(f"Artifact: {artifact_path}")
    print(f"Metrics: {metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
