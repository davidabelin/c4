"""Agent protocol definitions for c4 heuristic/model policies."""

from __future__ import annotations

from typing import Any, Protocol


class AgentCallable(Protocol):
    """Kaggle-compatible callable signature for Connect4 agents."""

    def __call__(self, obs: Any, config: Any) -> int:
        """Return the selected move column index."""

        ...
