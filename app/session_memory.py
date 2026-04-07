"""In-memory conversation state for short-lived chat sessions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from time import time


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


@dataclass
class _SessionState:
    turns: deque[ConversationTurn] = field(default_factory=deque)
    updated_at: float = field(default_factory=time)


class SessionMemoryStore:
    """Keep a bounded, expiring conversation history per session id."""

    def __init__(self, max_turns: int = 8, ttl_seconds: int = 3600) -> None:
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, _SessionState] = {}
        self._lock = Lock()

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        with self._lock:
            self._prune_expired_locked()
            state = self._sessions.get(session_id)
            if state is None:
                return []
            state.updated_at = time()
            return list(state.turns)

    def append_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        with self._lock:
            self._prune_expired_locked()
            state = self._sessions.setdefault(session_id, _SessionState())
            state.turns.append(ConversationTurn(role="user", content=user_message))
            state.turns.append(
                ConversationTurn(role="assistant", content=assistant_message)
            )
            while len(state.turns) > self._max_turns:
                state.turns.popleft()
            state.updated_at = time()

    def _prune_expired_locked(self) -> None:
        now = time()
        expired = [
            session_id
            for session_id, state in self._sessions.items()
            if now - state.updated_at > self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)
