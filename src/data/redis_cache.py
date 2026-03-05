"""Redis cache wrapper with JSON serialization."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

import redis

from src.core.settings import settings

log = logging.getLogger(__name__)


class RedisCache:
    """Thin wrapper around ``redis.Redis`` with JSON helpers."""

    def __init__(self, url: str | None = None) -> None:
        self._url = url or settings.redis_url
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis.from_url(
                self._url, decode_responses=True, socket_connect_timeout=5
            )
        return self._client

    # --- JSON helpers --------------------------------------------------------

    def get_json(self, key: str) -> Any:
        try:
            raw = self.client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError) as exc:
            log.debug("redis get_json(%s) failed: %s", key, exc)
            return None

    def set_json(
        self, key: str, value: Any, ttl_seconds: int = 3600
    ) -> None:
        try:
            self.client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
        except redis.RedisError as exc:
            log.debug("redis set_json(%s) failed: %s", key, exc)

    # --- Primitive helpers ---------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        try:
            return self.client.get(key)
        except redis.RedisError:
            return None

    def set(self, key: str, value: str, ttl_seconds: int = 3600) -> None:
        try:
            self.client.set(key, value, ex=ttl_seconds)
        except redis.RedisError:
            pass

    def delete(self, key: str) -> None:
        try:
            self.client.delete(key)
        except redis.RedisError:
            pass

    # --- Set helpers (atomic add/membership/read for event tracking) -------

    def sadd(self, key: str, *members: str, ttl_seconds: int = 0) -> int:
        """Atomically add members to a Redis set.  Returns count of new members."""
        try:
            added = self.client.sadd(key, *members) if members else 0
            if ttl_seconds > 0 and added:
                self.client.expire(key, ttl_seconds)
            return added
        except redis.RedisError as exc:
            log.debug("redis sadd(%s) failed: %s", key, exc)
            return 0

    def sismember(self, key: str, member: str) -> bool:
        try:
            return bool(self.client.sismember(key, member))
        except redis.RedisError:
            return False

    def smembers(self, key: str) -> set:
        try:
            return self.client.smembers(key)
        except redis.RedisError:
            return set()

    # --- Atomic lock helpers (SETNX) -----------------------------------------

    def setnx(self, key: str, value: str, ttl_seconds: int) -> bool:
        """Atomically set *key* only if it does not already exist.

        Uses Redis ``SET … NX EX`` (atomic set-if-not-exists with expiry).
        Returns ``True`` if the lock was acquired, ``False`` if it already
        existed (another worker holds it).
        """
        try:
            return bool(self.client.set(key, value, nx=True, ex=ttl_seconds))
        except redis.RedisError as exc:
            log.debug("redis setnx(%s) failed: %s", key, exc)
            return False


# Singleton instance used across the application
cache = RedisCache()
