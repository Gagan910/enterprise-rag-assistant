
import secrets
import time
from typing import Any

import requests

from src.auth.session_store import RedisSessionStore
from src.config.settings import settings


class SessionAuthenticationError(RuntimeError):
    """Raised when a Supabase session cannot be authenticated."""


class SupabaseAccessTokenInvalid(SessionAuthenticationError):
    """Raised when Supabase rejects an access token as invalid or expired."""


def validate_supabase_access_token(access_token: str) -> dict[str, Any]:
    """Validate an access token using Supabase Auth."""

    if not settings.supabase_url or not settings.supabase_anon_key:
        raise SessionAuthenticationError(
            "Supabase authentication is not configured."
        )

    try:
        response = requests.get(
            f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
            headers={
                "apikey": settings.supabase_anon_key,
                "Authorization": f"Bearer {access_token}",
            },
            timeout=10,
        )
    except requests.RequestException as exc:
        raise SessionAuthenticationError(
            "Unable to validate Supabase authentication."
        ) from exc

    if response.status_code in (401, 403):
        raise SupabaseAccessTokenInvalid(
            "Invalid or expired Supabase access token."
        )

    if not response.ok:
        raise SessionAuthenticationError(
            "Supabase authentication service is temporarily unavailable."
        )

    try:
        user = response.json()
    except ValueError as exc:
        raise SessionAuthenticationError(
            "Invalid Supabase authentication response."
        ) from exc

    if (
        not isinstance(user, dict)
        or not isinstance(user.get("id"), str)
        or not user["id"].strip()
    ):
        raise SessionAuthenticationError(
            "Supabase user identity is missing."
        )

    return user


def create_session(
    store: RedisSessionStore,
    access_token: str,
    refresh_token: str,
) -> tuple[str, dict[str, Any]]:
    """Validate Supabase credentials and create a server-side session."""

    if not access_token or not refresh_token:
        raise SessionAuthenticationError(
            "Session credentials are missing."
        )

    user = validate_supabase_access_token(access_token)
    session_id = store.new_session_id()

    store.create(
        session_id,
        {
            "user_id": user["id"],
            "access_token": access_token,
            "refresh_token": refresh_token,
            "access_token_checked_at": int(time.time()),
        },
    )

    return session_id, {
        "user_id": user["id"],
        "email": user.get("email"),
    }


def get_session(
    store: RedisSessionStore,
    session_id: str,
) -> dict[str, Any] | None:
    """Retrieve a session and refresh its access token when rejected."""

    if not session_id:
        return None

    record = store.get(session_id)

    if not record:
        return None

    access_token = record.get("access_token")
    refresh_token = record.get("refresh_token")
    user_id = record.get("user_id")

    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or not isinstance(user_id, str)
        or not user_id
    ):
        store.revoke(session_id)
        return None

    try:
        user = validate_supabase_access_token(access_token)
    except SupabaseAccessTokenInvalid:
        # An expired/rejected access token does not by itself mean that
        # the refresh token is invalid. Attempt a refresh before revoking.
        return refresh_session(store, session_id)

    if user.get("id") != user_id:
        store.revoke(session_id)
        return None

    return {
        "user_id": user_id,
        "email": user.get("email"),
    }


def revoke_session(
    store: RedisSessionStore,
    session_id: str,
) -> None:
    """Revoke a server-side session."""

    if session_id:
        store.revoke(session_id)


def refresh_session(
    store: RedisSessionStore,
    session_id: str,
) -> dict[str, Any] | None:
    """Refresh Supabase credentials under a distributed per-session lock."""

    lock_token = secrets.token_urlsafe(24)
    deadline = time.monotonic() + 5

    while not store.acquire_refresh_lock(session_id, lock_token):
        if time.monotonic() >= deadline:
            raise SessionAuthenticationError(
                "Session refresh is busy. Please retry."
            )
        time.sleep(0.1)

    try:
        # Read the latest record after acquiring the lock because another
        # request may have refreshed the tokens while this request waited.
        record = store.get(session_id)

        if not record:
            return None

        access_token = record.get("access_token")
        refresh_token = record.get("refresh_token")
        user_id = record.get("user_id")

        if (
            not isinstance(access_token, str)
            or not access_token
            or not isinstance(refresh_token, str)
            or not refresh_token
            or not isinstance(user_id, str)
            or not user_id
        ):
            store.revoke(session_id)
            return None

        if not settings.supabase_url or not settings.supabase_anon_key:
            raise SessionAuthenticationError(
                "Supabase authentication is not configured."
            )

        try:
            response = requests.post(
                f"{settings.supabase_url.rstrip('/')}/auth/v1/token",
                params={"grant_type": "refresh_token"},
                headers={"apikey": settings.supabase_anon_key},
                json={"refresh_token": refresh_token},
                timeout=10,
            )
        except requests.RequestException as exc:
            # Network errors are transient; preserve the session for retry.
            raise SessionAuthenticationError(
                "Unable to refresh Supabase authentication."
            ) from exc

        if not response.ok:
            if response.status_code in (400, 401, 403):
                # Supabase rejected the refresh credentials.
                store.revoke(session_id)
                return None

            # Do not destroy a potentially valid session on a Supabase
            # outage, rate limit, or server error.
            raise SessionAuthenticationError(
                "Supabase refresh service is temporarily unavailable."
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise SessionAuthenticationError(
                "Invalid Supabase refresh response."
            ) from exc

        if not isinstance(payload, dict):
            raise SessionAuthenticationError(
                "Invalid Supabase refresh response."
            )

        new_access_token = payload.get("access_token")
        new_refresh_token = payload.get("refresh_token")
        user = payload.get("user")

        if (
            not isinstance(new_access_token, str)
            or not new_access_token
            or not isinstance(new_refresh_token, str)
            or not new_refresh_token
            or not isinstance(user, dict)
            or user.get("id") != user_id
        ):
            store.revoke(session_id)
            return None

        updated_record = {
            **record,
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "access_token_checked_at": int(time.time()),
        }

        # Compare-and-update prevents overwriting credentials rotated by
        # another request. Redis performs this comparison atomically.
        if not store.update_if_refresh_token_matches(
            session_id,
            refresh_token,
            updated_record,
        ):
            # The stored session may have been revoked or changed.
            return None

        return {
            "user_id": user_id,
            "email": user.get("email"),
        }

    finally:
        store.release_refresh_lock(session_id, lock_token)
