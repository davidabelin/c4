"""Asynchronous orchestration for supervised training jobs."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from c4_storage.object_store import is_gcs_uri, join_storage_path
from c4_storage.repository import C4Repository
from c4_training.supervised import TrainConfig, train_model

try:
    from google.cloud import tasks_v2
except Exception:  # pragma: no cover
    tasks_v2 = None


class TrainingJobManager:
    """Manage supervised training execution locally or via Cloud Tasks."""

    def __init__(
        self,
        repository: C4Repository,
        models_dir: str,
        max_workers: int = 2,
        execution_mode: str = "local",
        task_project_id: str | None = None,
        task_location: str | None = None,
        task_queue: str | None = None,
        worker_url: str | None = None,
        worker_token: str | None = None,
        worker_service_account: str | None = None,
    ) -> None:
        self.repository = repository
        self.models_dir = str(models_dir)
        if not is_gcs_uri(self.models_dir):
            Path(self.models_dir).mkdir(parents=True, exist_ok=True)
        self.execution_mode = str(execution_mode).strip().lower()
        self.task_project_id = task_project_id
        self.task_location = task_location
        self.task_queue = task_queue
        self.worker_url = worker_url
        self.worker_token = worker_token
        self.worker_service_account = worker_service_account
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="c4-train")

    def submit_job(self, payload: dict) -> dict:
        """Persist and schedule a training job."""

        config = self._config_from_payload(payload)
        job = self.repository.create_training_job(config.model_type, payload)
        job_id = int(job["id"])
        if self.execution_mode == "task_queue":
            try:
                self._enqueue_job(job_id)
            except Exception as exc:
                self.repository.update_training_job(
                    job_id,
                    status="failed",
                    progress=1.0,
                    error_message=f"Queue enqueue failed: {exc}",
                )
        else:
            self.executor.submit(self._run_job, job_id, config)
        return self.repository.get_training_job(job_id) or job

    def _config_from_payload(self, payload: dict) -> TrainConfig:
        hidden_layer_sizes = payload.get("hidden_layer_sizes", [64, 32])
        if isinstance(hidden_layer_sizes, str):
            hidden_layer_sizes = [int(token.strip()) for token in hidden_layer_sizes.split(",") if token.strip()]
        return TrainConfig(
            model_type=str(payload.get("model_type", "decision_tree")),
            lookback=int(payload.get("lookback", 5)),
            test_size=float(payload.get("test_size", 0.2)),
            learning_rate=float(payload.get("learning_rate", 0.001)),
            hidden_layer_sizes=tuple(int(value) for value in hidden_layer_sizes),
            epochs=int(payload.get("epochs", payload.get("max_iter", 200))),
            batch_size=(
                "auto"
                if str(payload.get("batch_size", "auto")).strip().lower() == "auto"
                else int(payload.get("batch_size"))
            ),
            random_state=int(payload.get("random_state", 42)),
        )

    def _enqueue_job(self, job_id: int) -> None:
        if tasks_v2 is None:
            raise RuntimeError("google-cloud-tasks is required for task_queue execution mode")
        if not self.task_project_id or not self.task_location or not self.task_queue or not self.worker_url:
            raise RuntimeError("Cloud Tasks config is incomplete (project/location/queue/worker_url)")

        client = tasks_v2.CloudTasksClient()
        parent = client.queue_path(self.task_project_id, self.task_location, self.task_queue)
        url = f"{self.worker_url.rstrip('/')}/api/v1/internal/training/jobs/{job_id}/run"
        headers = {"Content-Type": "application/json"}
        if self.worker_token:
            headers["X-Worker-Token"] = self.worker_token
        request_payload = json.dumps({"job_id": job_id}).encode("utf-8")
        task: dict = {
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": url,
                "headers": headers,
                "body": request_payload,
            }
        }
        if self.worker_service_account:
            task["http_request"]["oidc_token"] = {
                "service_account_email": self.worker_service_account,
            }
        client.create_task(request={"parent": parent, "task": task})

    def run_job_by_id(self, job_id: int) -> None:
        """Execute a queued training job synchronously by id."""

        job = self.repository.get_training_job(job_id)
        if job is None:
            raise KeyError(f"Training job not found: {job_id}")
        if job["status"] in {"running", "completed", "failed"}:
            return
        params = json.loads(job["params_json"]) if job.get("params_json") else {}
        config = self._config_from_payload(params)
        self._run_job(job_id, config)

    def _run_job(self, job_id: int, config: TrainConfig) -> None:
        try:
            self.repository.update_training_job(job_id, status="running", progress=0.05)
            rows = self.repository.list_ai_moves_for_training()
            self.repository.update_training_job(job_id, progress=0.25)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            artifact_name = f"{config.model_type}_job{job_id}_{timestamp}.pkl"
            artifact_path = join_storage_path(self.models_dir, artifact_name)
            metrics = train_model(rows=rows, config=config, artifact_path=artifact_path)
            self.repository.update_training_job(job_id, progress=0.85, metrics=metrics)
            model_record = self.repository.create_model(
                name=f"{config.model_type}-{timestamp}",
                model_type=config.model_type,
                artifact_path=artifact_path,
                lookback=config.lookback,
                metrics=metrics,
            )
            self.repository.update_training_job(
                job_id,
                status="completed",
                progress=1.0,
                metrics=metrics,
                model_id=int(model_record["id"]),
            )
        except Exception as exc:  # pragma: no cover
            self.repository.update_training_job(job_id, status="failed", progress=1.0, error_message=str(exc))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
