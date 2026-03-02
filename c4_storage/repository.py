"""Repository layer for c4 metadata using SQLite/PostgreSQL via SQLAlchemy."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, MappingResult
from sqlalchemy.exc import DatabaseError


def utcnow_iso() -> str:
    """Return timezone-aware UTC timestamp string."""

    return datetime.now(UTC).isoformat()


def _looks_like_database_url(value: str) -> bool:
    """Best-effort check for SQLAlchemy-style DB URLs."""

    return "://" in str(value)


def _to_sqlite_url(path_value: str) -> str:
    """Convert filesystem path into SQLAlchemy SQLite URL."""

    path = Path(path_value).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+pysqlite:///{path.as_posix()}"


class C4Repository:
    """Persistence facade for c4 gameplay, training jobs, and model metadata."""

    def __init__(self, db_target: str) -> None:
        self._lock = RLock()
        target = str(db_target).strip()
        if not target:
            raise ValueError("Database target must not be empty.")
        self.db_target = target
        self.db_url = target if _looks_like_database_url(target) else _to_sqlite_url(target)
        connect_args = {"check_same_thread": False} if self.db_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(self.db_url, future=True, pool_pre_ping=True, connect_args=connect_args)
        self._dialect = self.engine.dialect.name

    def _run_script(self, script: str) -> None:
        statements = [part.strip() for part in script.split(";") if part.strip()]
        with self.engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))

    @staticmethod
    def _first_or_none(rows: MappingResult) -> dict | None:
        row = rows.first()
        return dict(row) if row is not None else None

    def init_schema(self) -> None:
        """Create required tables/indexes for the active dialect."""

        sqlite_schema = """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_name TEXT NOT NULL,
                session_index INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                winner TEXT,
                rounds_played INTEGER NOT NULL DEFAULT 0,
                score_player INTEGER NOT NULL DEFAULT 0,
                score_ai INTEGER NOT NULL DEFAULT 0,
                score_ties INTEGER NOT NULL DEFAULT 0,
                current_board_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS moves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id INTEGER NOT NULL,
                session_index INTEGER NOT NULL,
                move_index INTEGER NOT NULL,
                actor TEXT NOT NULL,
                action INTEGER NOT NULL,
                board_before_json TEXT NOT NULL,
                board_after_json TEXT NOT NULL,
                outcome TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(game_id) REFERENCES games(id)
            );

            CREATE TABLE IF NOT EXISTS training_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                model_type TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                metrics_json TEXT,
                error_message TEXT,
                model_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                lookback INTEGER NOT NULL,
                metrics_json TEXT,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(model_id) REFERENCES models(id)
            );

            CREATE TABLE IF NOT EXISTS rl_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                metrics_json TEXT,
                error_message TEXT,
                model_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_moves_game_session ON moves(game_id, session_index, move_index);
            CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_rl_jobs_status ON rl_jobs(status);
        """

        postgres_schema = """
            CREATE TABLE IF NOT EXISTS games (
                id BIGSERIAL PRIMARY KEY,
                agent_name TEXT NOT NULL,
                session_index INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                winner TEXT,
                rounds_played INTEGER NOT NULL DEFAULT 0,
                score_player INTEGER NOT NULL DEFAULT 0,
                score_ai INTEGER NOT NULL DEFAULT 0,
                score_ties INTEGER NOT NULL DEFAULT 0,
                current_board_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS moves (
                id BIGSERIAL PRIMARY KEY,
                game_id BIGINT NOT NULL REFERENCES games(id),
                session_index INTEGER NOT NULL,
                move_index INTEGER NOT NULL,
                actor TEXT NOT NULL,
                action INTEGER NOT NULL,
                board_before_json TEXT NOT NULL,
                board_after_json TEXT NOT NULL,
                outcome TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_jobs (
                id BIGSERIAL PRIMARY KEY,
                status TEXT NOT NULL,
                model_type TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                metrics_json TEXT,
                error_message TEXT,
                model_id BIGINT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS models (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                artifact_path TEXT NOT NULL,
                lookback INTEGER NOT NULL,
                metrics_json TEXT,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS model_metrics (
                id BIGSERIAL PRIMARY KEY,
                model_id BIGINT NOT NULL REFERENCES models(id),
                metric_name TEXT NOT NULL,
                metric_value DOUBLE PRECISION NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS rl_jobs (
                id BIGSERIAL PRIMARY KEY,
                status TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                metrics_json TEXT,
                error_message TEXT,
                model_id BIGINT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_moves_game_session ON moves(game_id, session_index, move_index);
            CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_rl_jobs_status ON rl_jobs(status);
        """

        with self._lock:
            self._run_script(sqlite_schema if self._dialect == "sqlite" else postgres_schema)

    def _insert_and_fetch(self, insert_sql: str, params: dict[str, Any], table_name: str) -> dict:
        returning_sql = f"{insert_sql} RETURNING *"
        with self._lock:
            with self.engine.begin() as conn:
                try:
                    row = self._first_or_none(conn.execute(text(returning_sql), params).mappings())
                    if row is not None:
                        return row
                except DatabaseError:
                    pass
                result = conn.execute(text(insert_sql), params)
                last_id = getattr(result, "lastrowid", None)
                if last_id is not None:
                    row = self._first_or_none(
                        conn.execute(text(f"SELECT * FROM {table_name} WHERE id = :id"), {"id": int(last_id)}).mappings()
                    )
                    if row is not None:
                        return row
                row = self._first_or_none(conn.execute(text(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1")).mappings())
                return row or {}

    def create_game(self, agent_name: str, board: list[int]) -> dict:
        """Create a new game row initialized to active state and empty score."""

        now = utcnow_iso()
        return self._insert_and_fetch(
            """
            INSERT INTO games (
                agent_name, session_index, status, winner, rounds_played,
                score_player, score_ai, score_ties, current_board_json, created_at, updated_at
            )
            VALUES (
                :agent_name, 0, 'active', NULL, 0, 0, 0, 0, :current_board_json, :created_at, :updated_at
            )
            """,
            {
                "agent_name": str(agent_name),
                "current_board_json": json.dumps(board),
                "created_at": now,
                "updated_at": now,
            },
            "games",
        )

    def get_game(self, game_id: int) -> dict | None:
        with self.engine.begin() as conn:
            return self._first_or_none(conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": int(game_id)}).mappings())

    def reset_game(self, game_id: int, board: list[int]) -> dict | None:
        game = self.get_game(game_id)
        if not game:
            return None
        now = utcnow_iso()
        new_session_index = int(game["session_index"]) + 1
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE games
                        SET session_index = :session_index,
                            status = 'active',
                            winner = NULL,
                            rounds_played = 0,
                            score_player = 0,
                            score_ai = 0,
                            score_ties = 0,
                            current_board_json = :current_board_json,
                            updated_at = :updated_at
                        WHERE id = :id
                        """
                    ),
                    {
                        "session_index": new_session_index,
                        "current_board_json": json.dumps(board),
                        "updated_at": now,
                        "id": int(game_id),
                    },
                )
                row = self._first_or_none(conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": int(game_id)}).mappings())
        return row

    def _next_move_index(self, conn, *, game_id: int, session_index: int) -> int:
        row = self._first_or_none(
            conn.execute(
                text(
                    """
                    SELECT MAX(move_index) AS max_idx
                    FROM moves
                    WHERE game_id = :game_id AND session_index = :session_index
                    """
                ),
                {"game_id": int(game_id), "session_index": int(session_index)},
            ).mappings()
        )
        max_idx = row["max_idx"] if row and row.get("max_idx") is not None else -1
        return int(max_idx) + 1

    def _insert_move_row(
        self,
        conn,
        *,
        game_id: int,
        session_index: int,
        move_index: int,
        actor: str,
        action: int,
        board_before: list[int],
        board_after: list[int],
        outcome: str,
        created_at: str,
    ) -> dict:
        try:
            row = self._first_or_none(
                conn.execute(
                    text(
                        """
                        INSERT INTO moves (
                            game_id, session_index, move_index, actor, action,
                            board_before_json, board_after_json, outcome, created_at
                        )
                        VALUES (
                            :game_id, :session_index, :move_index, :actor, :action,
                            :board_before_json, :board_after_json, :outcome, :created_at
                        )
                        RETURNING *
                        """
                    ),
                    {
                        "game_id": int(game_id),
                        "session_index": int(session_index),
                        "move_index": int(move_index),
                        "actor": str(actor),
                        "action": int(action),
                        "board_before_json": json.dumps(board_before),
                        "board_after_json": json.dumps(board_after),
                        "outcome": str(outcome),
                        "created_at": created_at,
                    },
                ).mappings()
            )
            if row is not None:
                return row
        except DatabaseError:
            pass

        result = conn.execute(
            text(
                """
                INSERT INTO moves (
                    game_id, session_index, move_index, actor, action,
                    board_before_json, board_after_json, outcome, created_at
                )
                VALUES (
                    :game_id, :session_index, :move_index, :actor, :action,
                    :board_before_json, :board_after_json, :outcome, :created_at
                )
                """
            ),
            {
                "game_id": int(game_id),
                "session_index": int(session_index),
                "move_index": int(move_index),
                "actor": str(actor),
                "action": int(action),
                "board_before_json": json.dumps(board_before),
                "board_after_json": json.dumps(board_after),
                "outcome": str(outcome),
                "created_at": created_at,
            },
        )
        last_id = getattr(result, "lastrowid", None)
        if last_id is not None:
            row = self._first_or_none(conn.execute(text("SELECT * FROM moves WHERE id = :id"), {"id": int(last_id)}).mappings())
            return row or {}
        row = self._first_or_none(conn.execute(text("SELECT * FROM moves ORDER BY id DESC LIMIT 1")).mappings())
        return row or {}

    def record_turn_and_update_game(
        self,
        *,
        game_id: int,
        session_index: int,
        player_action: int,
        ai_action: int | None,
        board_before_player: list[int],
        board_after_player: list[int],
        board_after_ai: list[int] | None,
        outcome: str,
    ) -> tuple[list[dict], dict]:
        """Persist one player turn (+ optional ai response) and update game state."""

        now = utcnow_iso()
        is_terminal = str(outcome) in {"player", "ai", "tie"}
        board_final = list(board_after_ai) if board_after_ai is not None else list(board_after_player)

        score_player_inc = 1 if str(outcome) == "player" else 0
        score_ai_inc = 1 if str(outcome) == "ai" else 0
        score_tie_inc = 1 if str(outcome) == "tie" else 0

        with self._lock:
            with self.engine.begin() as conn:
                current = self._first_or_none(
                    conn.execute(
                        text("SELECT * FROM games WHERE id = :id AND session_index = :session_index"),
                        {"id": int(game_id), "session_index": int(session_index)},
                    ).mappings()
                )
                if current is None:
                    raise KeyError(f"Game {game_id} not found or session mismatch.")

                move_index = self._next_move_index(conn, game_id=game_id, session_index=session_index)
                inserts: list[dict] = []

                player_row = self._insert_move_row(
                    conn,
                    game_id=game_id,
                    session_index=session_index,
                    move_index=move_index,
                    actor="player",
                    action=int(player_action),
                    board_before=board_before_player,
                    board_after=board_after_player,
                    outcome=(outcome if ai_action is None else "ongoing"),
                    created_at=now,
                )
                inserts.append(player_row)

                if ai_action is not None and board_after_ai is not None:
                    ai_row = self._insert_move_row(
                        conn,
                        game_id=game_id,
                        session_index=session_index,
                        move_index=move_index + 1,
                        actor="ai",
                        action=int(ai_action),
                        board_before=board_after_player,
                        board_after=board_after_ai,
                        outcome=outcome,
                        created_at=now,
                    )
                    inserts.append(ai_row)

                conn.execute(
                    text(
                        """
                        UPDATE games
                        SET rounds_played = rounds_played + 1,
                            score_player = score_player + :score_player_inc,
                            score_ai = score_ai + :score_ai_inc,
                            score_ties = score_ties + :score_tie_inc,
                            status = :status,
                            winner = :winner,
                            current_board_json = :current_board_json,
                            updated_at = :updated_at
                        WHERE id = :id AND session_index = :session_index
                        """
                    ),
                    {
                        "score_player_inc": int(score_player_inc),
                        "score_ai_inc": int(score_ai_inc),
                        "score_tie_inc": int(score_tie_inc),
                        "status": ("completed" if is_terminal else "active"),
                        "winner": (outcome if is_terminal else None),
                        "current_board_json": json.dumps(board_final),
                        "updated_at": now,
                        "id": int(game_id),
                        "session_index": int(session_index),
                    },
                )
                updated = self._first_or_none(
                    conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": int(game_id)}).mappings()
                )
                return inserts, (updated or {})

    def list_moves(self, game_id: int, session_index: int | None = None) -> list[dict]:
        query = "SELECT * FROM moves WHERE game_id = :game_id"
        params: dict[str, Any] = {"game_id": int(game_id)}
        if session_index is not None:
            query += " AND session_index = :session_index"
            params["session_index"] = int(session_index)
        query += " ORDER BY move_index ASC, id ASC"
        with self.engine.begin() as conn:
            rows = conn.execute(text(query), params).mappings().all()
        return [dict(row) for row in rows]

    def list_ai_moves_for_training(self) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT game_id, session_index, move_index, actor, action, board_before_json, board_after_json, outcome, created_at
                    FROM moves
                    WHERE actor = 'ai'
                    ORDER BY game_id ASC, session_index ASC, move_index ASC
                    """
                )
            ).mappings().all()
        return [dict(row) for row in rows]

    def create_training_job(self, model_type: str, params: dict) -> dict:
        now = utcnow_iso()
        return self._insert_and_fetch(
            """
            INSERT INTO training_jobs
            (status, model_type, params_json, progress, metrics_json, error_message, model_id, created_at, updated_at)
            VALUES ('queued', :model_type, :params_json, 0.0, NULL, NULL, NULL, :created_at, :updated_at)
            """,
            {"model_type": model_type, "params_json": json.dumps(params, sort_keys=True), "created_at": now, "updated_at": now},
            "training_jobs",
        )

    def update_training_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        progress: float | None = None,
        metrics: dict | None = None,
        error_message: str | None = None,
        model_id: int | None = None,
    ) -> dict | None:
        updates: list[str] = []
        params: dict[str, Any] = {}
        if status is not None:
            updates.append("status = :status")
            params["status"] = status
        if progress is not None:
            updates.append("progress = :progress")
            params["progress"] = float(progress)
        if metrics is not None:
            updates.append("metrics_json = :metrics_json")
            params["metrics_json"] = json.dumps(metrics, sort_keys=True)
        if error_message is not None:
            updates.append("error_message = :error_message")
            params["error_message"] = error_message
        if model_id is not None:
            updates.append("model_id = :model_id")
            params["model_id"] = int(model_id)
        updates.append("updated_at = :updated_at")
        params["updated_at"] = utcnow_iso()
        params["id"] = int(job_id)
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(text(f"UPDATE training_jobs SET {', '.join(updates)} WHERE id = :id"), params)
                row = self._first_or_none(
                    conn.execute(text("SELECT * FROM training_jobs WHERE id = :id"), {"id": int(job_id)}).mappings()
                )
        return row

    def get_training_job(self, job_id: int) -> dict | None:
        with self.engine.begin() as conn:
            return self._first_or_none(
                conn.execute(text("SELECT * FROM training_jobs WHERE id = :id"), {"id": int(job_id)}).mappings()
            )

    def list_training_jobs(self, limit: int = 100) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT * FROM training_jobs ORDER BY id DESC LIMIT :limit"), {"limit": int(limit)}).mappings().all()
        return [dict(row) for row in rows]

    def create_model(self, name: str, model_type: str, artifact_path: str, lookback: int, metrics: dict) -> dict:
        now = utcnow_iso()
        with self._lock:
            with self.engine.begin() as conn:
                row = None
                try:
                    row = self._first_or_none(
                        conn.execute(
                            text(
                                """
                                INSERT INTO models (name, model_type, artifact_path, lookback, metrics_json, is_active, created_at)
                                VALUES (:name, :model_type, :artifact_path, :lookback, :metrics_json, 0, :created_at)
                                RETURNING *
                                """
                            ),
                            {
                                "name": str(name),
                                "model_type": str(model_type),
                                "artifact_path": str(artifact_path),
                                "lookback": int(lookback),
                                "metrics_json": json.dumps(metrics, sort_keys=True),
                                "created_at": now,
                            },
                        ).mappings()
                    )
                except DatabaseError:
                    row = None
                if row is None:
                    result = conn.execute(
                        text(
                            """
                            INSERT INTO models (name, model_type, artifact_path, lookback, metrics_json, is_active, created_at)
                            VALUES (:name, :model_type, :artifact_path, :lookback, :metrics_json, 0, :created_at)
                            """
                        ),
                        {
                            "name": str(name),
                            "model_type": str(model_type),
                            "artifact_path": str(artifact_path),
                            "lookback": int(lookback),
                            "metrics_json": json.dumps(metrics, sort_keys=True),
                            "created_at": now,
                        },
                    )
                    model_id = int(getattr(result, "lastrowid", 0))
                    if model_id <= 0:
                        latest = self._first_or_none(conn.execute(text("SELECT * FROM models ORDER BY id DESC LIMIT 1")).mappings())
                        model_id = int(latest["id"]) if latest else 0
                    row = self._first_or_none(conn.execute(text("SELECT * FROM models WHERE id = :id"), {"id": model_id}).mappings())
                model_id = int(row["id"]) if row else 0
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        conn.execute(
                            text(
                                """
                                INSERT INTO model_metrics (model_id, metric_name, metric_value, created_at)
                                VALUES (:model_id, :metric_name, :metric_value, :created_at)
                                """
                            ),
                            {
                                "model_id": int(model_id),
                                "metric_name": str(key),
                                "metric_value": float(value),
                                "created_at": now,
                            },
                        )
        return row or {}

    def list_models(self, limit: int = 200) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT * FROM models ORDER BY id DESC LIMIT :limit"), {"limit": int(limit)}).mappings().all()
        return [dict(row) for row in rows]

    def get_model(self, model_id: int) -> dict | None:
        with self.engine.begin() as conn:
            return self._first_or_none(conn.execute(text("SELECT * FROM models WHERE id = :id"), {"id": int(model_id)}).mappings())

    def get_active_model(self) -> dict | None:
        with self.engine.begin() as conn:
            return self._first_or_none(
                conn.execute(text("SELECT * FROM models WHERE is_active = 1 ORDER BY id DESC LIMIT 1")).mappings()
            )

    def activate_model(self, model_id: int) -> dict | None:
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(text("UPDATE models SET is_active = 0"))
                conn.execute(text("UPDATE models SET is_active = 1 WHERE id = :id"), {"id": int(model_id)})
                row = self._first_or_none(conn.execute(text("SELECT * FROM models WHERE id = :id"), {"id": int(model_id)}).mappings())
        return row

    def create_rl_job(self, params: dict) -> dict:
        now = utcnow_iso()
        return self._insert_and_fetch(
            """
            INSERT INTO rl_jobs
            (status, params_json, progress, metrics_json, error_message, model_id, created_at, updated_at)
            VALUES ('queued', :params_json, 0.0, NULL, NULL, NULL, :created_at, :updated_at)
            """,
            {"params_json": json.dumps(params, sort_keys=True), "created_at": now, "updated_at": now},
            "rl_jobs",
        )

    def update_rl_job(
        self,
        job_id: int,
        *,
        status: str | None = None,
        progress: float | None = None,
        metrics: dict | None = None,
        error_message: str | None = None,
        model_id: int | None = None,
    ) -> dict | None:
        updates: list[str] = []
        named: dict[str, Any] = {}
        if status is not None:
            updates.append("status = :status")
            named["status"] = status
        if progress is not None:
            updates.append("progress = :progress")
            named["progress"] = float(progress)
        if metrics is not None:
            updates.append("metrics_json = :metrics_json")
            named["metrics_json"] = json.dumps(metrics, sort_keys=True)
        if error_message is not None:
            updates.append("error_message = :error_message")
            named["error_message"] = error_message
        if model_id is not None:
            updates.append("model_id = :model_id")
            named["model_id"] = int(model_id)
        updates.append("updated_at = :updated_at")
        named["updated_at"] = utcnow_iso()
        named["id"] = int(job_id)
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(text(f"UPDATE rl_jobs SET {', '.join(updates)} WHERE id = :id"), named)
                row = self._first_or_none(conn.execute(text("SELECT * FROM rl_jobs WHERE id = :id"), {"id": int(job_id)}).mappings())
        return row

    def get_rl_job(self, job_id: int) -> dict | None:
        with self.engine.begin() as conn:
            return self._first_or_none(conn.execute(text("SELECT * FROM rl_jobs WHERE id = :id"), {"id": int(job_id)}).mappings())

    def list_rl_jobs(self, limit: int = 100) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(text("SELECT * FROM rl_jobs ORDER BY id DESC LIMIT :limit"), {"limit": int(limit)}).mappings().all()
        return [dict(row) for row in rows]
