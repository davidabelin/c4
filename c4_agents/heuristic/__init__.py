"""Registry and factory functions for heuristic c4 agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from c4_agents.base import AgentCallable
from c4_agents.heuristic.adaptive_midrange import my_agent as adaptive_midrange
from c4_agents.heuristic.alpha_beta_v9 import my_agent as alpha_beta_v9
from c4_agents.heuristic.time_boxed_pruner import my_agent as time_boxed_pruner


@dataclass(frozen=True)
class AgentSpec:
    """Metadata descriptor for one registered heuristic agent."""

    name: str
    description: str
    factory: Callable[[], AgentCallable]


def _build_specs() -> dict[str, AgentSpec]:
    return {
        "alpha_beta_v9": AgentSpec(
            "alpha_beta_v9",
            "Depth-adaptive alpha-beta baseline migrated from legacy notebooks.",
            lambda: alpha_beta_v9,
        ),
        "adaptive_midrange": AgentSpec(
            "adaptive_midrange",
            "Mid-game weighted heuristic with adaptive depth.",
            lambda: adaptive_midrange,
        ),
        "time_boxed_pruner": AgentSpec(
            "time_boxed_pruner",
            "Alpha-beta search with time budget checks for Kaggle-like constraints.",
            lambda: time_boxed_pruner,
        ),
    }


AGENT_SPECS = _build_specs()


def list_agent_specs() -> list[AgentSpec]:
    """Return all registered specs sorted by name."""

    return [AGENT_SPECS[name] for name in sorted(AGENT_SPECS.keys())]


def build_heuristic_agent(name: str) -> AgentCallable:
    """Instantiate one heuristic agent callable by registry name."""

    if name not in AGENT_SPECS:
        raise KeyError(f"Unknown heuristic agent: {name}")
    return AGENT_SPECS[name].factory()


__all__ = [
    "AgentSpec",
    "AGENT_SPECS",
    "list_agent_specs",
    "build_heuristic_agent",
]
