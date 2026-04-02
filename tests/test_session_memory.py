from app.session_memory import SessionMemoryStore


def test_session_memory_store_keeps_recent_turns_only():
    store = SessionMemoryStore(max_turns=4, ttl_seconds=3600)

    store.append_exchange("session-1", "u1", "a1")
    store.append_exchange("session-1", "u2", "a2")
    store.append_exchange("session-1", "u3", "a3")

    history = store.get_history("session-1")

    assert [turn.content for turn in history] == ["u2", "a2", "u3", "a3"]


def test_session_memory_store_returns_empty_history_for_unknown_session():
    store = SessionMemoryStore()

    assert store.get_history("missing") == []