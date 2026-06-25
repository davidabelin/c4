"""Flask application factory and extension wiring for the c4 web app.

Role
----
Assemble the standalone Connect4 lab by wiring persistence, gameplay runtime
cache, supervised training jobs, RL jobs, arena jobs, and the page/API
blueprints into one Flask application.
"""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask

from c4_rl.jobs import RLJobManager
from c4_storage.repository import C4Repository
from c4_storage.sqlite_snapshot import SQLiteSnapshotMirror
from c4_training.jobs import TrainingJobManager
from c4_web.match_jobs import MatchJobManager
from c4_web.runtime import GameRuntimeCache


def _read_secret_version(secret_version_name: str) -> str:
    """Read one secret payload from Secret Manager by full version resource name."""

    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": secret_version_name})
    return response.payload.data.decode("utf-8")


def _resolve_secret_into_config(app: Flask, *, target_key: str, source_key: str) -> None:
    """Populate a config value from Secret Manager when direct value is empty.

    This lets local and cloud deployments use either direct env values or
    Secret Manager indirection.
    """

    if str(app.config.get(target_key, "")).strip():
        return
    secret_version_name = str(app.config.get(source_key, "")).strip()
    if not secret_version_name:
        return
    try:
        app.config[target_key] = _read_secret_version(secret_version_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed loading {target_key} from Secret Manager secret version '{secret_version_name}': {exc}"
        ) from exc


def create_app(config: dict | None = None) -> Flask:
    """Create and configure the standalone Flask application instance."""

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_mapping(
        SECRET_KEY="dev-only-secret-key-change-me",
        DATABASE_URL=os.getenv("C4_DATABASE_URL", ""),
        DATABASE_URL_SECRET=os.getenv("C4_DATABASE_URL_SECRET", ""),
        DB_PATH=os.getenv("C4_DB_PATH", str(data_dir / "c4.db")),
        DB_SNAPSHOT_URI=os.getenv("C4_DB_SNAPSHOT_URI", ""),
        EVENTS_DIR=os.getenv("C4_EVENTS_DIR", str(data_dir / "events")),
        MODELS_DIR=os.getenv("C4_MODELS_DIR", str(data_dir / "models")),
        EXPORTS_DIR=os.getenv("C4_EXPORTS_DIR", str(data_dir / "exports")),
        DEFAULT_AGENT="alpha_beta_v9",
        TRAINING_EXECUTION_MODE=os.getenv("C4_TRAINING_EXECUTION_MODE", "local"),
        TASKS_PROJECT_ID=os.getenv("C4_TASKS_PROJECT_ID", ""),
        TASKS_LOCATION=os.getenv("C4_TASKS_LOCATION", ""),
        TASKS_QUEUE=os.getenv("C4_TASKS_QUEUE", ""),
        TRAINING_WORKER_URL=os.getenv("C4_TRAINING_WORKER_URL", ""),
        TRAINING_WORKER_TOKEN=os.getenv("C4_TRAINING_WORKER_TOKEN", ""),
        TRAINING_WORKER_TOKEN_SECRET=os.getenv("C4_TRAINING_WORKER_TOKEN_SECRET", ""),
        TRAINING_WORKER_SERVICE_ACCOUNT=os.getenv("C4_TRAINING_WORKER_SERVICE_ACCOUNT", ""),
        INTERNAL_WORKER_TOKEN=os.getenv("C4_INTERNAL_WORKER_TOKEN", ""),
        INTERNAL_WORKER_TOKEN_SECRET=os.getenv("C4_INTERNAL_WORKER_TOKEN_SECRET", ""),
        AGENT_MATCH_DEFAULT_TURNS=int(os.getenv("C4_AGENT_MATCH_DEFAULT_TURNS", "42")),
        DRL_HOME_URL=os.getenv("DRL_HOME_URL", "http://127.0.0.1:5000/"),
    )
    if config:
        app.config.update(config)

    _resolve_secret_into_config(app, target_key="DATABASE_URL", source_key="DATABASE_URL_SECRET")
    _resolve_secret_into_config(
        app,
        target_key="TRAINING_WORKER_TOKEN",
        source_key="TRAINING_WORKER_TOKEN_SECRET",
    )
    _resolve_secret_into_config(
        app,
        target_key="INTERNAL_WORKER_TOKEN",
        source_key="INTERNAL_WORKER_TOKEN_SECRET",
    )

    sqlite_snapshot = None
    if not app.config["DATABASE_URL"]:
        sqlite_snapshot = SQLiteSnapshotMirror(
            db_path=str(app.config["DB_PATH"]),
            snapshot_uri=str(app.config["DB_SNAPSHOT_URI"]),
            logger=app.logger,
        )
        sqlite_snapshot.download_if_missing()

    database_target = app.config["DATABASE_URL"] or app.config["DB_PATH"]
    repository = C4Repository(database_target)
    repository.init_schema()
    if sqlite_snapshot is not None:
        sqlite_snapshot.sync_after_schema_init()
        app.extensions["sqlite_snapshot"] = sqlite_snapshot

        @app.after_request
        def mirror_sqlite_snapshot(response):
            sqlite_snapshot.upload_if_changed()
            return response

    training_jobs = TrainingJobManager(
        repository,
        models_dir=app.config["MODELS_DIR"],
        execution_mode=app.config["TRAINING_EXECUTION_MODE"],
        task_project_id=app.config["TASKS_PROJECT_ID"] or None,
        task_location=app.config["TASKS_LOCATION"] or None,
        task_queue=app.config["TASKS_QUEUE"] or None,
        worker_url=app.config["TRAINING_WORKER_URL"] or None,
        worker_token=(app.config["TRAINING_WORKER_TOKEN"] or app.config["INTERNAL_WORKER_TOKEN"] or None),
        worker_service_account=app.config["TRAINING_WORKER_SERVICE_ACCOUNT"] or None,
    )
    app.extensions["repository"] = repository
    app.extensions["training_jobs"] = training_jobs
    app.extensions["game_runtime"] = GameRuntimeCache(max_entries=256)
    app.extensions["rl_jobs"] = RLJobManager(repository, models_dir=app.config["MODELS_DIR"])
    app.extensions["match_jobs"] = MatchJobManager(repository, default_agent=app.config["DEFAULT_AGENT"])

    @app.context_processor
    def inject_shared_links() -> dict[str, str]:
        """Expose parent-lab navigation without coupling this app to DRL routes."""

        return {"drl_home_url": str(app.config.get("DRL_HOME_URL", "")).strip()}

    from c4_web.blueprints.arena import arena_bp
    from c4_web.blueprints.game import game_bp
    from c4_web.blueprints.main import main_bp
    from c4_web.blueprints.rl import rl_bp
    from c4_web.blueprints.training import training_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(game_bp)
    app.register_blueprint(arena_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(rl_bp)

    return app
