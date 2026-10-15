
import json
from unittest.mock import MagicMock

import pytest
from redis.exceptions import RedisError

from src.auth.session_store import (
    RedisSessionStore,
    SessionStoreUnavailable,
)


@pytest.fixture
def store():
    instance = RedisSessionStore(
        "redis://localhost:6379/0",
        ttl_seconds=300,
    )
    instance._client = MagicMock()
    return instance


def test_new_session_id_is_random_and_url_safe():
    first = RedisSessionStore.new_session_id()
    second = RedisSessionStore.new_session_id()

    assert first != second
    assert len(first) >= 40
    assert all(c.isalnum() or c in "-_" for c in first)


def test_create_stores_record_with_ttl_and_unique_key(store):
    store._client.set.return_value = True
    session_id = "opaque-session-id"
    record = {"user_id": "user-123"}

    store.create(session_id, record)

    args, kwargs = store._client.set.call_args
    assert args[0].startswith("rag:session:")
    assert json.loads(args[1]) == record
    assert kwargs["ex"] == 300
    assert kwargs["nx"] is True


def test_create_rejects_duplicate_session(store):
    store._client.set.return_value = None

    with pytest.raises(RuntimeError, match="unique session"):
        store.create("session-id", {"user_id": "user-123"})


def test_get_returns_decoded_record(store):
    record = {"user_id": "user-123"}
    store._client.get.return_value = json.dumps(record)

    assert store.get("session-id") == record


def test_get_returns_none_for_missing_session(store):
    store._client.get.return_value = None

    assert store.get("session-id") is None


def test_get_rejects_invalid_record(store):
    store._client.get.return_value = "not-json"

    with pytest.raises(SessionStoreUnavailable):
        store.get("session-id")


def test_update_only_updates_existing_session(store):
    store._client.set.return_value = True

    updated = store.update(
        "session-id",
        {"user_id": "user-123"},
    )

    assert updated is True
    assert store._client.set.call_args.kwargs["xx"] is True
    assert store._client.set.call_args.kwargs["ex"] == 300


def test_update_returns_false_for_missing_session(store):
    store._client.set.return_value = None

    assert store.update("session-id", {"user_id": "user-123"}) is False


def test_revoke_deletes_session(store):
    store.revoke("session-id")

    store._client.delete.assert_called_once()
    assert store._client.delete.call_args.args[0].startswith(
        "rag:session:"
    )


def test_redis_errors_fail_closed(store):
    store._client.get.side_effect = RedisError("connection failed")

    with pytest.raises(SessionStoreUnavailable):
        store.get("session-id")
