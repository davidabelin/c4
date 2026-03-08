"""Asynchronous orchestration for persisted agent-vs-agent Connect4 matches."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from random import Random

from c4_agents import ModelBackedAgent, build_heuristic_agent, list_agent_specs
from c4_core.matches import play_agent_match
from c4_core.types import Connect4Config
from c4_storage.repository import C4Repository


def _available_agent_names() -> list[str]:
    return [spec.name for spec in list_agent_specs()]


def _default_match_opponent(agent_name: str) -> str:
    for candidate in _available_agent_names():
        if candidate != agent_name:
            return candidate
    return agent_name


def _build_agent_from_name(repository: C4Repository, agent_name: str):
    if agent_name == "active_model":
        model_record = repository.get_active_model()
        if model_record is None:
            raise RuntimeError("No active model is available. Activate a trained model first.")
        return ModelBackedAgent(str(model_record["artifact_path"]))
    if agent_name not in set(_available_agent_names()):
        raise KeyError(f"Unknown agent '{agent_name}'.")
    return build_heuristic_agent(agent_name)


class MatchJobManager:
    """Manage background arena matches and persisted replay traces."""

    def __init__(self, repository: C4Repository, default_agent: str, max_workers: int = 2) -> None:
        self.repository = repository
        self.default_agent = str(default_agent)
        self.config = Connect4Config(rows=6, columns=7, inarow=4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="c4-arena")

    def submit_job(self, payload: dict) -> dict:
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
        if starting_agent == "random":
            starting_agent = "agent_a" if Random(seed).random() < 0.5 else "agent_b"
        return {
            "agent_a": agent_a,
            "agent_b": agent_b,
            "starting_agent": starting_agent,
            "max_turns": max_turns,
            "seed": seed,
        }

    def _run_job(self, job_id: int, config: dict) -> None:
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
                    "moves_played": 0,
                    "winner": None,
                    "status": "running",
                },
                trace=[],
            )

            agent_a = _build_agent_from_name(self.repository, config["agent_a"])
            agent_b = _build_agent_from_name(self.repository, config["agent_b"])

            def _on_move(frame: dict) -> None:
                trace.append(frame)
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
                trace=match["trace"],
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
        self.executor.shutdown(wait=False)
