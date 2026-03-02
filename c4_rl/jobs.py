"""Asynchronous orchestration for RL training jobs."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from c4_rl.trainer import QTrainConfig, train_q_table
from c4_storage.object_store import is_gcs_uri, join_storage_path, write_bytes
from c4_storage.repository import C4Repository


class RLJobManager:
    """Manage RL job submission and lifecycle persistence."""

    def __init__(self, repository: C4Repository, models_dir: str, max_workers: int = 1) -> None:
        self.repository = repository
        self.models_dir = str(models_dir)
        if not is_gcs_uri(self.models_dir):
            Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="c4-rl")

    def submit_job(self, payload: dict) -> dict:
        config = self._config_from_payload(payload)
        job = self.repository.create_rl_job(payload)
        job_id = int(job["id"])
        self.executor.submit(self._run_job, job_id, config)
        return job

    @staticmethod
    def _config_from_payload(payload: dict) -> QTrainConfig:
        return QTrainConfig(
            episodes=int(payload.get("episodes", 300)),
            alpha=float(payload.get("alpha", 0.1)),
            gamma=float(payload.get("gamma", 0.6)),
            epsilon_start=float(payload.get("epsilon_start", 0.99)),
            epsilon_end=float(payload.get("epsilon_end", 0.1)),
            epsilon_decay=float(payload.get("epsilon_decay", 0.9999)),
            alpha_decay_step=int(payload.get("alpha_decay_step", 1000)),
            alpha_decay_rate=float(payload.get("alpha_decay_rate", 0.9)),
            opponent=str(payload.get("opponent", "negamax")),
            switch_prob=float(payload.get("switch_prob", 0.5)),
            seed=int(payload.get("seed", 7)),
        )

    def _run_job(self, job_id: int, config: QTrainConfig) -> None:
        try:
            self.repository.update_rl_job(job_id, status="running", progress=0.05)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            artifact_name = f"rl_qtable_job{job_id}_{timestamp}.pkl"
            artifact_path = join_storage_path(self.models_dir, artifact_name)
            self.repository.update_rl_job(job_id, progress=0.25)
            artifact = train_q_table(config=config)
            payload = pickle.dumps(artifact)
            write_bytes(artifact_path, payload, content_type="application/octet-stream")
            metrics = dict(artifact.get("metrics", {}))
            metrics["artifact_path"] = artifact_path
            self.repository.update_rl_job(job_id, progress=0.85, metrics=metrics)
            model_row = self.repository.create_model(
                name=f"rl-qtable-{timestamp}",
                model_type="rl_qtable",
                artifact_path=artifact_path,
                lookback=1,
                metrics=metrics,
            )
            self.repository.update_rl_job(
                job_id,
                status="completed",
                progress=1.0,
                metrics=metrics,
                model_id=int(model_row["id"]),
            )
        except Exception as exc:  # pragma: no cover
            self.repository.update_rl_job(job_id, status="failed", progress=1.0, error_message=str(exc))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
