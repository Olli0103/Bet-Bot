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


# Singleton instance used across the application
cache = RedisCache()
