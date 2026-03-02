"""In-memory runtime cache for low-latency c4 gameplay sessions."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any


@dataclass(slots=True)
class GameRuntimeState:
    """Cached per-game runtime state."""

    game_id: int
    agent_name: str
    session_index: int
    signature: str
    agent: Any


class GameRuntimeCache:
    """Thread-safe LRU-style cache of ``GameRuntimeState`` objects."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max(16, int(max_entries))
        self._lock = RLock()
        self._items: OrderedDict[int, GameRuntimeState] = OrderedDict()

    def get(self, game_id: int) -> GameRuntimeState | None:
        with self._lock:
            state = self._items.get(int(game_id))
            if state is None:
                return None
            self._items.move_to_end(int(game_id))
            return state

    def put(self, state: GameRuntimeState) -> None:
        key = int(state.game_id)
        with self._lock:
            self._items[key] = state
            self._items.move_to_end(key)
            while len(self._items) > self.max_entries:
                self._items.popitem(last=False)

    def forget_game(self, game_id: int) -> None:
        with self._lock:
            self._items.pop(int(game_id), None)
