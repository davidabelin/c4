"""Asynchronous orchestration for persisted agent-vs-agent Connect4 matches.

Role
----
Own the persisted arena workflow for Connect4, including optional per-move
analysis snapshots that enrich the replay trace for the spectator UI.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from random import Random

from c4_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from c4_core.forecast import forecast_columns
from c4_core.matches import play_agent_match
from c4_core.types import Connect4Config
from c4_storage.repository import C4Repository


def _available_agent_names() -> list[str]:
    """Return all heuristic and built-in Connect4 agent names."""

    return [spec.name for spec in list_agent_specs()]


def _default_match_opponent(agent_name: str) -> str:
    """Choose a default opponent distinct from the requested agent."""

    for candidate in _available_agent_names():
        if candidate != agent_name:
            return candidate
    return agent_name


def _build_agent_from_name(repository: C4Repository, agent_name: str):
    """Resolve one arena-facing agent identifier into a concrete agent object."""

    if agent_name == "active_model":
        model_record = repository.get_active_model()
        if model_record is None:
            raise RuntimeError("No active model is available. Activate a trained model first.")
        return ModelBackedAgent(str(model_record["artifact_path"]))
    if agent_name not in set(_available_agent_names()):
        raise KeyError(f"Unknown agent '{agent_name}'.")
    return build_heuristic_agent(agent_name)


class MatchJobManager:
    """Manage background arena matches and persisted replay traces.

    Cross-Repo Context
    ------------------
    This is the Connect4 parallel to ``rps_web.match_jobs.MatchJobManager``,
    with one extra responsibility: enriching replay frames with heuristic column
    forecasts for the arena viewer.
    """

    def __init__(self, repository: C4Repository, default_agent: str, max_workers: int = 2) -> None:
        self.repository = repository
        self.default_agent = str(default_agent)
        self.config = Connect4Config(rows=6, columns=7, inarow=4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="c4-arena")

    def submit_job(self, payload: dict) -> dict:
        """Validate one arena payload, persist the match row, and enqueue work."""

        config = self._config_from_payload(payload)
        _build_agent_from_name(self.repository, config["agent_a"])
        _build_agent_from_name(self.repository, config["agent_b"])
        job = self.repository.create_arena_match(
            agent_a_name=config["agent_a"],
            agent_b_name=config["agent_b"],
            params=config,
        )
        self.executor.submit(self._run_job, int(job["id"]), config)
        return job

    def _config_from_payload(self, payload: dict) -> dict:
        """Normalize one API payload into the stored arena match config."""

        agent_a = str(payload.get("agent_a", self.default_agent))
        agent_b = str(payload.get("agent_b", _default_match_opponent(agent_a)))
        starting_agent = str(payload.get("starting_agent", "agent_a")).strip().lower()
        if starting_agent not in {"agent_a", "agent_b", "random"}:
            raise ValueError("starting_agent must be one of: agent_a, agent_b, random.")
        raw_seed = payload.get("seed")
        seed = int(raw_seed) if raw_seed is not None else None
        raw_max_turns = payload.get("max_turns")
        max_turns = int(raw_max_turns) if raw_max_turns is not None else int(self.config.rows) * int(self.config.columns)
        if max_turns <= 0 or max_turns > int(self.config.rows) * int(self.config.columns):
            raise ValueError("max_turns must be between 1 and 42.")
        analysis_enabled = bool(payload.get("analysis_enabled", True))
        raw_analysis_lookahead = payload.get("analysis_lookahead")
        analysis_lookahead = int(raw_analysis_lookahead) if raw_analysis_lookahead is not None else 4
        analysis_lookahead = max(1, min(6, analysis_lookahead))
        if starting_agent == "random":
            starting_agent = "agent_a" if Random(seed).random() < 0.5 else "agent_b"
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "starting_agent": starting_agent,
            "max_turns": max_turns,
            "seed": seed,
            "analysis_enabled": analysis_enabled,
            "analysis_lookahead": analysis_lookahead,
        }

    def _run_job(self, job_id: int, config: dict) -> None:
        """Execute one persisted Connect4 arena match and update replay state."""

        trace: list[dict] = []
        max_turns = int(config["max_turns"])
        try:
            self.repository.update_arena_match(
                job_id,
                status="running",
                progress=0.01,
                summary={
                    "mode": "agent_vs_agent",
                    "agent_a": config["agent_a"],
                    "agent_b": config["agent_b"],
                    "starting_agent": config["starting_agent"],
                    "max_turns": max_turns,
                    "seed": config["seed"],
                    "analysis_enabled": bool(config["analysis_enabled"]),
                    "analysis_lookahead": int(config["analysis_lookahead"]),
                    "moves_played": 0,
                    "winner": None,
                    "status": "running",
                },
                trace=[],
            )

            agent_a = _build_agent_from_name(self.repository, config["agent_a"])
            agent_b = _build_agent_from_name(self.repository, config["agent_b"])

            def _on_move(frame: dict) -> None:
                enriched = dict(frame)
                if bool(config["analysis_enabled"]):
                    forecasts = forecast_columns(
                        list(frame["board_before"]),
                        perspective_mark=int(frame["mark"]),
                        lookahead=int(config["analysis_lookahead"]),
                        config=self.config,
                    )
                    enriched["forecasts"] = forecasts
                    enriched["recommended_column"] = forecasts[0]["column"] if forecasts else None
                    enriched["analysis_lookahead"] = int(config["analysis_lookahead"])
                trace.append(enriched)
                self.repository.update_arena_match(
                    job_id,
                    status="running",
                    progress=len(trace) / max(1, max_turns),
                    summary={
                        "mode": "agent_vs_agent",
                        "agent_a": config["agent_a"],
                        "agent_b": config["agent_b"],
                        "starting_agent": config["starting_agent"],
                        "max_turns": max_turns,
                        "seed": config["seed"],
                        "analysis_enabled": bool(config["analysis_enabled"]),
                        "analysis_lookahead": int(config["analysis_lookahead"]),
                        "moves_played": len(trace),
                        "winner": None,
                        "status": "running",
                    },
                    trace=list(trace),
                )

            match = play_agent_match(
                agent_a=agent_a,
                agent_b=agent_b,
                agent_a_name=config["agent_a"],
                agent_b_name=config["agent_b"],
                config=self.config,
                starting_agent=config["starting_agent"],
                max_turns=max_turns,
                seed=config["seed"],
                on_move=_on_move,
            )
            summary = {key: value for key, value in match.items() if key != "trace"}
            self.repository.update_arena_match(
                job_id,
                status="completed",
                progress=1.0,
                winner=(str(match["winner"]) if match["winner"] is not None else None),
                summary=summary,
                trace=list(trace),
            )
        except Exception as exc:  # pragma: no cover
            self.repository.update_arena_match(
                job_id,
                status="failed",
                progress=1.0,
                error_message=str(exc),
                trace=trace,
            )

    def shutdown(self) -> None:
        """Release local executor resources without waiting for active jobs."""

        self.executor.shutdown(wait=False)
