"""Reddit unauthenticated JSON fetcher for team sentiment posts.

Uses Reddit's unofficial ``*.json`` endpoints (no OAuth required).
Enforces strict rate-limiting (≤10 QPM / ≥6 s between requests) to
avoid IP bans on the unauthenticated API.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import List

import aiohttp

from src.data.redis_cache import cache
from src.integrations.base_fetcher import _safe_sync_run

log = logging.getLogger(__name__)

# ── Reddit rate-limit guard ──────────────────────────────────
# Unauthenticated Reddit API: max 10 requests per minute per IP.
# We enforce a minimum 6-second gap between ANY two Reddit requests
# to stay safely under that ceiling.
_MIN_REQUEST_INTERVAL = 6.0
_last_request_ts: float = 0.0

# IMPORTANT: Replace 'YOUR_REDDIT_USERNAME' with your actual Reddit
# username before deploying.  Reddit silently blocks requests that use
# a generic or missing User-Agent.
_USER_AGENT = (
    "python:bet-bot-sentiment-scraper:v1.0 (by /u/YOUR_REDDIT_USERNAME)"
)

_BASE_URL = "https://www.reddit.com"


class RedditFetcher:
    """Async fetcher for Reddit's public JSON endpoints.

    Designed to be used inside the existing ``asyncio`` event-loop
    architecture (e.g. from ``enrichment.py`` or ``core_worker.py``).

    Usage::

        fetcher = RedditFetcher()
        text = await fetcher.fetch_team_sentiment_posts("Bayern Munich")
        # → concatenated titles + selftext of the top 5 recent posts
        await fetcher.close()
    """

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    # -- session lifecycle ----------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create (or recreate) an ``aiohttp.ClientSession``."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": _USER_AGENT},
                timeout=self._timeout,
            )
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # -- rate-limiting --------------------------------------------------------

    @staticmethod
    async def _respect_rate_limit() -> None:
        """Sleep until at least ``_MIN_REQUEST_INTERVAL`` seconds have
        elapsed since the last Reddit request."""
        global _last_request_ts
        now = time.monotonic()
        elapsed = now - _last_request_ts
        if elapsed < _MIN_REQUEST_INTERVAL:
            wait = _MIN_REQUEST_INTERVAL - elapsed
            log.debug("reddit_rate_limit: sleeping %.1fs", wait)
            await asyncio.sleep(wait)
        _last_request_ts = time.monotonic()

    # -- public API -----------------------------------------------------------

    async def fetch_team_sentiment_posts(self, team_name: str, cache_ttl: int = 3600) -> str:
        """Search ``/r/sportsbook`` for recent posts mentioning *team_name*.

        Returns the concatenated ``title + selftext`` of up to 5 posts as
        a single string suitable for downstream LLM sentiment analysis.
        Returns an empty string on any error (never crashes the bot).

        Results are cached in Redis for *cache_ttl* seconds (default 1 h)
        to avoid unnecessary Reddit requests.
        """
        cache_key = f"reddit:posts:{team_name.lower()}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return str(cached)

        session = self._ensure_session()
        await self._respect_rate_limit()

        url = f"{_BASE_URL}/r/sportsbook/search.json"
        params = {
            "q": team_name,
            "restrict_sr": "1",
            "sort": "new",
            "limit": "5",
        }

        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    log.warning(
                        "reddit_rate_limited: HTTP 429 for team '%s' — "
                        "backing off, returning empty",
                        team_name,
                    )
                    return ""

                if resp.status != 200:
                    log.warning(
                        "reddit_fetch_error: HTTP %d for team '%s'",
                        resp.status,
                        team_name,
                    )
                    return ""

                data = await resp.json(content_type=None)

        except asyncio.TimeoutError:
            log.warning("reddit_timeout: request timed out for team '%s'", team_name)
            return ""
        except aiohttp.ClientError as exc:
            log.warning("reddit_client_error: %s for team '%s'", exc, team_name)
            return ""
        except Exception as exc:
            log.warning("reddit_unexpected_error: %s for team '%s'", exc, team_name)
            return ""

        text = self._parse_posts(data, team_name)
        cache.set_json(cache_key, text, cache_ttl)
        return text

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _parse_posts(data: dict, team_name: str) -> str:
        """Extract ``title`` + ``selftext`` from Reddit search JSON.

        Gracefully handles missing or malformed keys — Reddit's
        unauthenticated JSON schema sometimes omits ``selftext``.
        """
        try:
            children: List[dict] = data.get("data", {}).get("children", [])
        except (AttributeError, TypeError):
            log.debug("reddit_parse: unexpected payload structure for '%s'", team_name)
            return ""

        parts: List[str] = []
        for child in children[:5]:
            post = child.get("data", {})
            title = (post.get("title") or "").strip()
            body = (post.get("selftext") or "").strip()
            if title:
                parts.append(title)
            if body:
                parts.append(body)

        return "\n\n".join(parts)

    # -- sync convenience wrapper ---------------------------------------------

    def fetch_team_sentiment_posts_sync(self, team_name: str, cache_ttl: int = 3600) -> str:
        """Synchronous wrapper — safe to call from inside a running event loop.

        Ensures the ``aiohttp`` session is closed within the same event
        loop that created it, preventing ``ResourceWarning`` leaks.
        """
        async def _fetch_and_close() -> str:
            try:
                return await self.fetch_team_sentiment_posts(team_name, cache_ttl=cache_ttl)
            finally:
                await self.close()

        return _safe_sync_run(_fetch_and_close())
