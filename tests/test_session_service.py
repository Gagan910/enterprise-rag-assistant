
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.auth.session_service import (
    SessionAuthenticationError,
    create_session,
    get_session,
    refresh_session,
    revoke_session,
)
from src.auth.session_store import RedisSessionStore


@pytest.fixture
def store():
    return MagicMock(spec=RedisSessionStore)


@patch("src.auth.session_service.validate_supabase_access_token")
def test_create_session_validates_and_stores_credentials(
    validate_token,
    store,
):
    validate_token.return_value = {
        "id": "user-123",
        "email": "user@example.com",
    }
    store.new_session_id.return_value = "opaque-session-id"

    session_id, user = create_session(
        store,
        "access-token",
        "refresh-token",
    )

    assert session_id == "opaque-session-id"
    assert user["user_id"] == "user-123"
    store.create.assert_called_once()
    record = store.create.call_args.args[1]
    assert record["access_token"] == "access-token"
    assert record["refresh_token"] == "refresh-token"


def test_create_session_rejects_missing_credentials(store):
    with pytest.raises(SessionAuthenticationError):
        create_session(store, "", "refresh-token")


@patch("src.auth.session_service.validate_supabase_access_token")
def test_get_session_returns_public_user_data(validate_token, store):
    store.get.return_value = {
        "user_id": "user-123",
        "access_token": "access-token",
        "refresh_token": "refresh-token",
    }
    validate_token.return_value = {
        "id": "user-123",
        "email": "user@example.com",
    }

    result = get_session(store, "session-id")

    assert result == {
        "user_id": "user-123",
        "email": "user@example.com",
    }
    assert "access_token" not in result
    assert "refresh_token" not in result


def test_get_session_returns_none_when_missing(store):
    store.get.return_value = None

    assert get_session(store, "missing-session") is None


@patch("src.auth.session_service.requests.post")
@patch("src.auth.session_service.settings")
def test_refresh_session_rotates_tokens(settings_mock, post, store):
    settings_mock.supabase_url = "https://example.supabase.co"
    settings_mock.supabase_anon_key = "anon-key"

    store.acquire_refresh_lock.return_value = True
    store.get.return_value = {
        "user_id": "user-123",
        "access_token": "old-access",
        "refresh_token": "old-refresh",
    }
    store.update_if_refresh_token_matches.return_value = True

    response = MagicMock()
    response.ok = True
    response.json.return_value = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "user": {
            "id": "user-123",
            "email": "user@example.com",
        },
    }
    post.return_value = response

    result = refresh_session(store, "session-id")

    assert result["user_id"] == "user-123"
    store.update_if_refresh_token_matches.assert_called_once()
    updated = store.update_if_refresh_token_matches.call_args.args[2]
    assert updated["access_token"] == "new-access"
    assert updated["refresh_token"] == "new-refresh"
    store.release_refresh_lock.assert_called_once()


def test_revoke_session_deletes_session(store):
    revoke_session(store, "session-id")

    store.revoke.assert_called_once_with("session-id")
