
import hashlib
import secrets
from typing import Any

import redis
from redis.exceptions import RedisError


SESSION_TTL_SECONDS = 60 * 60 * 24 * 7
SESSION_KEY_PREFIX = "rag:session:"


class SessionStoreUnavailable(RuntimeError):
    """Raised when the session store cannot be reached."""


class RedisSessionStore:
    """Store server-side session records keyed by opaque session IDs."""

    def __init__(self, redis_url: str, ttl_seconds: int = SESSION_TTL_SECONDS):
        if not redis_url:
            raise ValueError("A Redis URL is required.")

        if ttl_seconds <= 0:
            raise ValueError("Session TTL must be positive.")

        self._client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            health_check_interval=30,
        )
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return f"{SESSION_KEY_PREFIX}{digest}"

    @staticmethod
    def new_session_id() -> str:
        """Generate a high-entropy opaque session identifier."""
        return secrets.token_urlsafe(32)

    def create(self, session_id: str, record: dict[str, Any]) -> None:
        """Create a session record with a fixed expiration."""
        try:
            created = self._client.set(
                self._key(session_id),
                __import__("json").dumps(record),
                ex=self._ttl_seconds,
                nx=True,
            )
        except RedisError as exc:
            raise SessionStoreUnavailable(
                "Session storage is unavailable."
            ) from exc

        if not created:
            raise RuntimeError("Unable to create a unique session.")

    def get(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session record without extending its lifetime."""
        try:
            raw = self._client.get(self._key(session_id))
        except RedisError as exc:
            raise SessionStoreUnavailable(
                "Session storage is unavailable."
            ) from exc

        if raw is None:
            return None

        try:
            import json
            record = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise SessionStoreUnavailable(
                "Stored session data is invalid."
            ) from exc

        if not isinstance(record, dict):
            raise SessionStoreUnavailable("Stored session data is invalid.")

        return record

    def update(self, session_id: str, record: dict[str, Any]) -> bool:
        """Update a session only if it still exists."""
        try:
            updated = self._client.set(
                self._key(session_id),
                __import__("json").dumps(record),
                ex=self._ttl_seconds,
                xx=True,
            )
        except RedisError as exc:
            raise SessionStoreUnavailable(
                "Session storage is unavailable."
            ) from exc

        return bool(updated)

    def update_if_refresh_token_matches(
        self,
        session_id: str,
        expected_refresh_token: str,
        record: dict[str, Any],
    ) -> bool:
        """Atomically update a session only if its refresh token is unchanged."""
        script = """
        local raw = redis.call('GET', KEYS[1])
        if not raw then
            return 0
        end

        local current = cjson.decode(raw)
        if current['refresh_token'] ~= ARGV[1] then
            return 0
        end

        redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
        return 1
        """

        import json

        try:
            result = self._client.eval(
                script,
                1,
                self._key(session_id),
                expected_refresh_token,
                json.dumps(record),
                self._ttl_seconds,
            )
        except RedisError as exc:
            raise SessionStoreUnavailable(
                "Session storage is unavailable."
            ) from exc

        return bool(result)


    def acquire_refresh_lock(
        self,
        session_id: str,
        lock_token: str,
        lock_ttl_seconds: int = 15,
    ) -> bool:
        """Acquire a short-lived distributed lock for session refresh."""
        try:
            return bool(
                self._client.set(
                    f"{self._key(session_id)}:refresh-lock",
                    lock_token,
                    ex=lock_ttl_seconds,
                    nx=True,
                )
            )
        except RedisError as exc:
            raise SessionStoreUnavailable(
                "Session storage is unavailable."
            ) from exc

    def release_refresh_lock(
        self,
        session_id: str,
        lock_token: str,
    ) -> None:
        """Release the lock only if this caller owns it."""
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """

        try:
            self._client.eval(
                script,
                1,
                f"{self._key(session_id)}:refresh-lock",
                lock_token,
            )
        except RedisError as exc:
            raise SessionStoreUnavailable(
                "Session storage is unavailable."
            ) from exc

    def revoke(self, session_id: str) -> None:
        """Delete a session record."""
        try:
            self._client.delete(self._key(session_id))
        except RedisError as exc:
            raise SessionStoreUnavailable(
                "Session storage is unavailable."
            ) from exc

    def close(self) -> None:
        self._client.close()
