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

    @staticmethod
    def _normalize_training_selection(selection: str | None) -> str | None:
        if selection is None:
            return None
        normalized = str(selection).strip().lower()
        if not normalized or normalized == "unset":
            return None
        if normalized not in {"include", "exclude"}:
            raise ValueError("selection must be one of: include, exclude, unset.")
        return normalized

    @staticmethod
    def _normalize_selection_mode(selection_mode: str | None) -> str:
        normalized = str(selection_mode or "all").strip().lower()
        if normalized not in {"all", "selected"}:
            raise ValueError("selection_mode must be one of: all, selected.")
        return normalized

    @staticmethod
    def _normalize_actor_scope(actor_scope: str | None) -> str:
        normalized = str(actor_scope or "algorithm").strip().lower()
        if normalized not in {"algorithm", "human", "all"}:
            raise ValueError("actor_scope must be one of: algorithm, human, all.")
        return normalized

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

            CREATE TABLE IF NOT EXISTS arena_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                agent_a_name TEXT NOT NULL,
                agent_b_name TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0.0,
                winner TEXT,
                summary_json TEXT,
                trace_json TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_session_selections (
                source_kind TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                session_index INTEGER NOT NULL,
                selection TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source_kind, source_id, session_index)
            );

            CREATE INDEX IF NOT EXISTS idx_moves_game_session ON moves(game_id, session_index, move_index);
            CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_rl_jobs_status ON rl_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_arena_matches_status ON arena_matches(status);
            CREATE INDEX IF NOT EXISTS idx_training_session_selections_selection
                ON training_session_selections(selection, source_kind, source_id, session_index);
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

            CREATE TABLE IF NOT EXISTS arena_matches (
                id BIGSERIAL PRIMARY KEY,
                status TEXT NOT NULL,
                agent_a_name TEXT NOT NULL,
                agent_b_name TEXT NOT NULL,
                params_json TEXT NOT NULL,
                progress DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                winner TEXT,
                summary_json TEXT,
                trace_json TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS training_session_selections (
                source_kind TEXT NOT NULL,
                source_id BIGINT NOT NULL,
                session_index INTEGER NOT NULL,
                selection TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source_kind, source_id, session_index)
            );

            CREATE INDEX IF NOT EXISTS idx_moves_game_session ON moves(game_id, session_index, move_index);
            CREATE INDEX IF NOT EXISTS idx_models_active ON models(is_active);
            CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_rl_jobs_status ON rl_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_arena_matches_status ON arena_matches(status);
            CREATE INDEX IF NOT EXISTS idx_training_session_selections_selection
                ON training_session_selections(selection, source_kind, source_id, session_index);
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

    def record_ai_opening_move(
        self,
        *,
        game_id: int,
        session_index: int,
        ai_action: int,
        board_before: list[int],
        board_after: list[int],
    ) -> tuple[dict, dict]:
        """Persist one AI opening move without incrementing player-turn counters."""

        now = utcnow_iso()
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
                ai_row = self._insert_move_row(
                    conn,
                    game_id=game_id,
                    session_index=session_index,
                    move_index=move_index,
                    actor="ai",
                    action=int(ai_action),
                    board_before=board_before,
                    board_after=board_after,
                    outcome="ongoing",
                    created_at=now,
                )
                conn.execute(
                    text(
                        """
                        UPDATE games
                        SET current_board_json = :current_board_json,
                            updated_at = :updated_at
                        WHERE id = :id AND session_index = :session_index
                        """
                    ),
                    {
                        "current_board_json": json.dumps(board_after),
                        "updated_at": now,
                        "id": int(game_id),
                        "session_index": int(session_index),
                    },
                )
                updated = self._first_or_none(
                    conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": int(game_id)}).mappings()
                )
                return ai_row, (updated or {})

    def undo_last_turn(self, *, game_id: int, session_index: int) -> tuple[dict, dict] | None:
        """Undo the latest move set (player+ai turn, or lone opening ai move)."""

        now = utcnow_iso()
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

                rows = conn.execute(
                    text(
                        """
                        SELECT * FROM moves
                        WHERE game_id = :game_id AND session_index = :session_index
                        ORDER BY move_index DESC, id DESC
                        """
                    ),
                    {"game_id": int(game_id), "session_index": int(session_index)},
                ).mappings().all()
                if not rows:
                    return None

                last = dict(rows[0])
                delete_ids: list[int] = []
                undo_summary = {
                    "kind": "",
                    "removed_player": False,
                    "removed_ai": False,
                    "terminal_outcome_removed": None,
                }
                if str(last.get("actor")) == "ai":
                    prev_player = self._first_or_none(
                        conn.execute(
                            text(
                                """
                                SELECT * FROM moves
                                WHERE game_id = :game_id
                                  AND session_index = :session_index
                                  AND move_index = :move_index
                                  AND actor = 'player'
                                LIMIT 1
                                """
                            ),
                            {
                                "game_id": int(game_id),
                                "session_index": int(session_index),
                                "move_index": int(last["move_index"]) - 1,
                            },
                        ).mappings()
                    )
                    if prev_player is not None:
                        delete_ids = [int(prev_player["id"]), int(last["id"])]
                        board_revert = json.loads(str(prev_player["board_before_json"]))
                        rounds_decrement = 1
                        terminal = str(last.get("outcome") or prev_player.get("outcome") or "")
                        undo_summary["kind"] = "turn"
                        undo_summary["removed_player"] = True
                        undo_summary["removed_ai"] = True
                    else:
                        delete_ids = [int(last["id"])]
                        board_revert = json.loads(str(last["board_before_json"]))
                        rounds_decrement = 0
                        terminal = str(last.get("outcome") or "")
                        undo_summary["kind"] = "opening_ai_move"
                        undo_summary["removed_ai"] = True
                else:
                    delete_ids = [int(last["id"])]
                    board_revert = json.loads(str(last["board_before_json"]))
                    rounds_decrement = 1
                    terminal = str(last.get("outcome") or "")
                    undo_summary["kind"] = "player_only_turn"
                    undo_summary["removed_player"] = True

                score_player_dec = 1 if terminal == "player" and rounds_decrement > 0 else 0
                score_ai_dec = 1 if terminal == "ai" and rounds_decrement > 0 else 0
                score_tie_dec = 1 if terminal == "tie" and rounds_decrement > 0 else 0
                undo_summary["terminal_outcome_removed"] = terminal if terminal in {"player", "ai", "tie"} else None

                if len(delete_ids) == 1:
                    conn.execute(
                        text("DELETE FROM moves WHERE id = :id"),
                        {"id": int(delete_ids[0])},
                    )
                else:
                    conn.execute(
                        text("DELETE FROM moves WHERE id = :id_one OR id = :id_two"),
                        {"id_one": int(delete_ids[0]), "id_two": int(delete_ids[1])},
                    )

                conn.execute(
                    text(
                        """
                        UPDATE games
                        SET rounds_played = CASE
                              WHEN rounds_played - :rounds_decrement < 0 THEN 0
                              ELSE rounds_played - :rounds_decrement
                            END,
                            score_player = CASE
                              WHEN score_player - :score_player_dec < 0 THEN 0
                              ELSE score_player - :score_player_dec
                            END,
                            score_ai = CASE
                              WHEN score_ai - :score_ai_dec < 0 THEN 0
                              ELSE score_ai - :score_ai_dec
                            END,
                            score_ties = CASE
                              WHEN score_ties - :score_tie_dec < 0 THEN 0
                              ELSE score_ties - :score_tie_dec
                            END,
                            status = 'active',
                            winner = NULL,
                            current_board_json = :current_board_json,
                            updated_at = :updated_at
                        WHERE id = :id AND session_index = :session_index
                        """
                    ),
                    {
                        "rounds_decrement": int(rounds_decrement),
                        "score_player_dec": int(score_player_dec),
                        "score_ai_dec": int(score_ai_dec),
                        "score_tie_dec": int(score_tie_dec),
                        "current_board_json": json.dumps(board_revert),
                        "updated_at": now,
                        "id": int(game_id),
                        "session_index": int(session_index),
                    },
                )
                updated = self._first_or_none(
                    conn.execute(text("SELECT * FROM games WHERE id = :id"), {"id": int(game_id)}).mappings()
                )
                undo_summary["removed_move_count"] = len(delete_ids)
                undo_summary["removed_move_ids"] = delete_ids
                undo_summary["rounds_decrement"] = int(rounds_decrement)
                return (updated or {}), undo_summary

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

    def _training_selection_lookup(self, conn) -> dict[tuple[str, int, int], str]:
        rows = conn.execute(
            text(
                """
                SELECT source_kind, source_id, session_index, selection
                FROM training_session_selections
                """
            )
        ).mappings().all()
        return {
            (str(row["source_kind"]), int(row["source_id"]), int(row["session_index"])): str(row["selection"])
            for row in rows
        }

    @staticmethod
    def _is_training_session_included(
        selection_lookup: dict[tuple[str, int, int], str],
        *,
        source_kind: str,
        source_id: int,
        session_index: int,
        selection_mode: str,
    ) -> bool:
        selection = selection_lookup.get((str(source_kind), int(source_id), int(session_index)))
        if selection_mode == "selected":
            return selection == "include"
        return selection != "exclude"

    @staticmethod
    def _actor_allowed_for_scope(actor: str, actor_scope: str) -> bool:
        actor_name = str(actor).strip().lower()
        if actor_scope == "all":
            return actor_name in {"player", "ai", "agent_a", "agent_b"}
        if actor_scope == "human":
            return actor_name == "player"
        return actor_name in {"ai", "agent_a", "agent_b"}

    def set_training_session_selection(
        self,
        *,
        source_kind: str,
        source_id: int,
        session_index: int,
        selection: str | None,
    ) -> dict:
        normalized = self._normalize_training_selection(selection)
        source_kind_text = str(source_kind).strip().lower()
        if source_kind_text not in {"game", "arena"}:
            raise ValueError("source_kind must be one of: game, arena.")
        source_id_value = int(source_id)
        session_index_value = int(session_index)
        now = utcnow_iso()
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        DELETE FROM training_session_selections
                        WHERE source_kind = :source_kind
                          AND source_id = :source_id
                          AND session_index = :session_index
                        """
                    ),
                    {
                        "source_kind": source_kind_text,
                        "source_id": source_id_value,
                        "session_index": session_index_value,
                    },
                )
                if normalized is not None:
                    conn.execute(
                        text(
                            """
                            INSERT INTO training_session_selections
                            (source_kind, source_id, session_index, selection, updated_at)
                            VALUES (:source_kind, :source_id, :session_index, :selection, :updated_at)
                            """
                        ),
                        {
                            "source_kind": source_kind_text,
                            "source_id": source_id_value,
                            "session_index": session_index_value,
                            "selection": normalized,
                            "updated_at": now,
                        },
                    )
        return {
            "source_kind": source_kind_text,
            "source_id": source_id_value,
            "session_index": session_index_value,
            "selection": normalized,
        }

    def delete_training_session(
        self,
        *,
        source_kind: str,
        source_id: int,
        session_index: int,
    ) -> dict:
        source_kind_text = str(source_kind).strip().lower()
        if source_kind_text not in {"game", "arena"}:
            raise ValueError("source_kind must be one of: game, arena.")

        source_id_value = int(source_id)
        session_index_value = int(session_index)
        deleted_moves = 0

        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        DELETE FROM training_session_selections
                        WHERE source_kind = :source_kind
                          AND source_id = :source_id
                          AND session_index = :session_index
                        """
                    ),
                    {
                        "source_kind": source_kind_text,
                        "source_id": source_id_value,
                        "session_index": session_index_value,
                    },
                )

                if source_kind_text == "game":
                    game_row = self._first_or_none(
                        conn.execute(
                            text("SELECT id, status, session_index FROM games WHERE id = :id"),
                            {"id": source_id_value},
                        ).mappings()
                    )
                    if game_row is None:
                        raise ValueError("Training session was not found.")
                    if (
                        str(game_row.get("status") or "").strip().lower() == "active"
                        and int(game_row.get("session_index", -1)) == session_index_value
                    ):
                        raise ValueError("Cannot delete the active gameplay session. Finish or reset it first.")

                    deleted_moves = int(
                        conn.execute(
                            text(
                                """
                                DELETE FROM moves
                                WHERE game_id = :game_id
                                  AND session_index = :session_index
                                """
                            ),
                            {
                                "game_id": source_id_value,
                                "session_index": session_index_value,
                            },
                        ).rowcount
                        or 0
                    )
                    if deleted_moves <= 0:
                        raise ValueError("Training session was not found.")
                else:
                    match_row = self._first_or_none(
                        conn.execute(
                            text("SELECT id, status FROM arena_matches WHERE id = :id"),
                            {"id": source_id_value},
                        ).mappings()
                    )
                    if match_row is None:
                        raise ValueError("Training session was not found.")
                    if str(match_row.get("status") or "").strip().lower() in {"queued", "running"}:
                        raise ValueError("Cannot delete an arena match while it is queued or running.")

                    deleted_matches = int(
                        conn.execute(
                            text("DELETE FROM arena_matches WHERE id = :id"),
                            {"id": source_id_value},
                        ).rowcount
                        or 0
                    )
                    if deleted_matches <= 0:
                        raise ValueError("Training session was not found.")
                    deleted_moves = deleted_matches

        return {
            "source_kind": source_kind_text,
            "source_id": source_id_value,
            "session_index": session_index_value,
            "deleted_rows": deleted_moves,
        }

    def list_training_sessions(self, limit: int = 200) -> list[dict]:
        limit_value = max(1, int(limit))
        with self.engine.begin() as conn:
            selection_lookup = self._training_selection_lookup(conn)
            gameplay_rows = conn.execute(
                text(
                    """
                    SELECT
                        m.game_id AS source_id,
                        m.session_index AS session_index,
                        m.move_index AS move_index,
                        m.actor AS actor,
                        m.outcome AS outcome,
                        m.created_at AS created_at,
                        g.agent_name AS opponent_name
                    FROM moves m
                    JOIN games g ON g.id = m.game_id
                    ORDER BY m.game_id ASC, m.session_index ASC, m.move_index ASC
                    """
                )
            ).mappings().all()
            arena_rows = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        status,
                        winner,
                        agent_a_name,
                        agent_b_name,
                        created_at,
                        updated_at,
                        trace_json
                    FROM arena_matches
                    WHERE trace_json IS NOT NULL
                    ORDER BY updated_at DESC, id DESC
                    """
                )
            ).mappings().all()

        sessions: list[dict] = []
        grouped_gameplay: dict[tuple[int, int], list[dict]] = {}
        for row in gameplay_rows:
            key = (int(row["source_id"]), int(row["session_index"]))
            grouped_gameplay.setdefault(key, []).append(dict(row))

        for (source_id, session_index), grouped_rows in grouped_gameplay.items():
            human_moves = sum(1 for row in grouped_rows if str(row["actor"]) == "player")
            algorithm_moves = sum(1 for row in grouped_rows if str(row["actor"]) == "ai")
            final_outcome = str(grouped_rows[-1].get("outcome") or "")
            status = "completed" if final_outcome in {"player", "ai", "tie"} else "active"
            winner = final_outcome if status == "completed" else None
            sessions.append(
                {
                    "source_kind": "game",
                    "source_id": source_id,
                    "session_index": session_index,
                    "session_type": "human_vs_algorithm",
                    "status": status,
                    "winner": winner,
                    "label": f"Game {source_id} / Session {session_index}",
                    "matchup_label": f"Human vs {grouped_rows[0]['opponent_name']}",
                    "opponent_name": grouped_rows[0]["opponent_name"],
                    "human_moves": int(human_moves),
                    "algorithm_moves": int(algorithm_moves),
                    "total_moves": int(len(grouped_rows)),
                    "started_at": grouped_rows[0]["created_at"],
                    "updated_at": grouped_rows[-1]["created_at"],
                    "selection": selection_lookup.get(("game", source_id, session_index)),
                }
            )

        for row in arena_rows:
            trace = json.loads(str(row["trace_json"])) if row.get("trace_json") else []
            if not isinstance(trace, list) or not trace:
                continue
            source_id = int(row["id"])
            session_index = 0
            sessions.append(
                {
                    "source_kind": "arena",
                    "source_id": source_id,
                    "session_index": session_index,
                    "session_type": "algorithm_vs_algorithm",
                    "status": row["status"],
                    "winner": row["winner"],
                    "label": f"Arena Match {source_id}",
                    "matchup_label": f"{row['agent_a_name']} vs {row['agent_b_name']}",
                    "opponent_name": None,
                    "agent_a_name": row["agent_a_name"],
                    "agent_b_name": row["agent_b_name"],
                    "human_moves": 0,
                    "algorithm_moves": len(trace),
                    "total_moves": len(trace),
                    "started_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "selection": selection_lookup.get(("arena", source_id, session_index)),
                }
            )

        sessions.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return sessions[:limit_value]

    def list_training_rows(
        self,
        *,
        selection_mode: str = "all",
        actor_scope: str = "algorithm",
    ) -> list[dict]:
        normalized_selection_mode = self._normalize_selection_mode(selection_mode)
        normalized_actor_scope = self._normalize_actor_scope(actor_scope)

        with self.engine.begin() as conn:
            selection_lookup = self._training_selection_lookup(conn)
            move_rows = conn.execute(
                text(
                    """
                    SELECT game_id, session_index, move_index, actor, action, board_before_json, board_after_json, outcome, created_at
                    FROM moves
                    ORDER BY game_id ASC, session_index ASC, move_index ASC
                    """
                )
            ).mappings().all()
            arena_rows = conn.execute(
                text(
                    """
                    SELECT id, created_at, updated_at, trace_json
                    FROM arena_matches
                    WHERE trace_json IS NOT NULL
                    ORDER BY id ASC
                    """
                )
            ).mappings().all()

        rows: list[dict] = []
        for row in move_rows:
            source_id = int(row["game_id"])
            session_index = int(row["session_index"])
            if not self._is_training_session_included(
                selection_lookup,
                source_kind="game",
                source_id=source_id,
                session_index=session_index,
                selection_mode=normalized_selection_mode,
            ):
                continue
            if not self._actor_allowed_for_scope(str(row["actor"]), normalized_actor_scope):
                continue
            payload = dict(row)
            payload["source_kind"] = "game"
            payload["source_id"] = source_id
            payload["actor_role"] = "human" if str(row["actor"]) == "player" else "algorithm"
            rows.append(payload)

        if normalized_actor_scope != "human":
            for row in arena_rows:
                source_id = int(row["id"])
                session_index = 0
                if not self._is_training_session_included(
                    selection_lookup,
                    source_kind="arena",
                    source_id=source_id,
                    session_index=session_index,
                    selection_mode=normalized_selection_mode,
                ):
                    continue
                trace = json.loads(str(row["trace_json"])) if row.get("trace_json") else []
                if not isinstance(trace, list):
                    continue
                for frame in trace:
                    rows.append(
                        {
                            "source_kind": "arena",
                            "source_id": source_id,
                            "game_id": source_id,
                            "session_index": session_index,
                            "move_index": int(frame["move_index"]),
                            "actor": str(frame["actor"]),
                            "actor_role": "algorithm",
                            "action": int(frame["action"]),
                            "board_before_json": json.dumps(frame["board_before"]),
                            "board_after_json": json.dumps(frame["board_after"]),
                            "outcome": frame.get("outcome"),
                            "created_at": row["updated_at"] or row["created_at"],
                        }
                    )

        rows.sort(
            key=lambda item: (
                str(item.get("source_kind") or "game"),
                int(item.get("source_id", item.get("game_id", 0))),
                int(item.get("session_index", 0)),
                int(item.get("move_index", 0)),
            )
        )
        return rows

    def list_ai_moves_for_training(
        self,
        *,
        selection_mode: str = "all",
        actor_scope: str = "algorithm",
    ) -> list[dict]:
        return self.list_training_rows(selection_mode=selection_mode, actor_scope=actor_scope)

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

    def create_arena_match(self, *, agent_a_name: str, agent_b_name: str, params: dict) -> dict:
        now = utcnow_iso()
        return self._insert_and_fetch(
            """
            INSERT INTO arena_matches
            (
                status, agent_a_name, agent_b_name, params_json, progress, winner,
                summary_json, trace_json, error_message, created_at, updated_at
            )
            VALUES (
                'queued', :agent_a_name, :agent_b_name, :params_json, 0.0, NULL,
                NULL, NULL, NULL, :created_at, :updated_at
            )
            """,
            {
                "agent_a_name": str(agent_a_name),
                "agent_b_name": str(agent_b_name),
                "params_json": json.dumps(params, sort_keys=True),
                "created_at": now,
                "updated_at": now,
            },
            "arena_matches",
        )

    def update_arena_match(
        self,
        match_id: int,
        *,
        status: str | None = None,
        progress: float | None = None,
        winner: str | None = None,
        summary: dict | None = None,
        trace: list[dict] | None = None,
        error_message: str | None = None,
    ) -> dict | None:
        updates: list[str] = []
        named: dict[str, Any] = {}
        if status is not None:
            updates.append("status = :status")
            named["status"] = status
        if progress is not None:
            updates.append("progress = :progress")
            named["progress"] = float(progress)
        if winner is not None:
            updates.append("winner = :winner")
            named["winner"] = winner
        if summary is not None:
            updates.append("summary_json = :summary_json")
            named["summary_json"] = json.dumps(summary, sort_keys=True)
        if trace is not None:
            updates.append("trace_json = :trace_json")
            named["trace_json"] = json.dumps(trace, sort_keys=True)
        if error_message is not None:
            updates.append("error_message = :error_message")
            named["error_message"] = error_message
        updates.append("updated_at = :updated_at")
        named["updated_at"] = utcnow_iso()
        named["id"] = int(match_id)
        with self._lock:
            with self.engine.begin() as conn:
                conn.execute(text(f"UPDATE arena_matches SET {', '.join(updates)} WHERE id = :id"), named)
                row = self._first_or_none(
                    conn.execute(text("SELECT * FROM arena_matches WHERE id = :id"), {"id": int(match_id)}).mappings()
                )
        return row

    def get_arena_match(self, match_id: int) -> dict | None:
        with self.engine.begin() as conn:
            return self._first_or_none(
                conn.execute(text("SELECT * FROM arena_matches WHERE id = :id"), {"id": int(match_id)}).mappings()
            )

    def list_arena_matches(self, limit: int = 100) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("SELECT * FROM arena_matches ORDER BY id DESC LIMIT :limit"),
                {"limit": int(limit)},
            ).mappings().all()
        return [dict(row) for row in rows]
