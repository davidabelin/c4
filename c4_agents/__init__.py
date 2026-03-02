"""Agent registry and factory functions for c4."""

from c4_agents.base import AgentCallable
from c4_agents.heuristic import AGENT_SPECS, AgentSpec, build_heuristic_agent, list_agent_specs
from c4_agents.model_agent import ModelBackedAgent

__all__ = [
    "AgentCallable",
    "AgentSpec",
    "AGENT_SPECS",
    "build_heuristic_agent",
    "list_agent_specs",
    "ModelBackedAgent",
]
