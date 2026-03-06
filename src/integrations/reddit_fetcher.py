"""Reddit RSS fetcher for team sentiment posts.

Primary path uses subreddit Atom/RSS feeds (``/r/<sub>/.rss``), which are
stable without OAuth and easy to curate by league/team coverage.
Falls back to legacy unauthenticated JSON listing endpoints when needed.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import List, Optional

import aiohttp

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import _safe_sync_run

log = logging.getLogger(__name__)

# ── Reddit rate-limit guard ──────────────────────────────────
# Unauthenticated Reddit API: max 10 requests per minute per IP.
# We enforce a minimum 6-second gap between ANY two Reddit requests
# to stay safely under that ceiling.
_MIN_REQUEST_INTERVAL = 6.0
_last_request_ts: float = 0.0

# User-Agent is configurable via REDDIT_USER_AGENT in settings/.env.
_USER_AGENT = settings.reddit_user_agent

_BASE_URL = "https://www.reddit.com"
_OLD_BASE_URL = "https://old.reddit.com"
_OAUTH_BASE_URL = "https://oauth.reddit.com"
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"


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
        self._oauth_token: Optional[str] = None
        self._oauth_expires_at: float = 0.0
        self._client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
        self._client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()

    # -- session lifecycle ----------------------------------------------------

    def _ensure_session(self) -> aiohttp.ClientSession:
        """Lazily create (or recreate) an ``aiohttp.ClientSession``.

        Uses unified SSL configuration from base_fetcher.
        """
        if self._session is None or self._session.closed:
            from src.integrations.base_fetcher import build_ssl_context
            ssl_ctx = build_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_ctx)
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": _USER_AGENT},
                timeout=self._timeout,
                connector=connector,
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

    async def _ensure_oauth_token(self) -> Optional[str]:
        """Fetch/refresh OAuth app token when credentials are configured."""
        if not self._client_id or not self._client_secret:
            return None

        now = time.time()
        if self._oauth_token and now < self._oauth_expires_at - 30:
            return self._oauth_token

        session = self._ensure_session()
        await self._respect_rate_limit()
        data = {"grant_type": "client_credentials"}
        auth = aiohttp.BasicAuth(self._client_id, self._client_secret)

        try:
            async with session.post(_TOKEN_URL, data=data, auth=auth) as resp:
                if resp.status != 200:
                    log.warning("reddit_oauth_token_error: HTTP %d", resp.status)
                    return None
                payload = await resp.json(content_type=None)
        except Exception as exc:
            log.warning("reddit_oauth_token_exception: %s", exc)
            return None

        token = payload.get("access_token")
        expires_in = int(payload.get("expires_in", 3600))
        if not token:
            return None
        self._oauth_token = str(token)
        self._oauth_expires_at = now + max(60, expires_in)
        return self._oauth_token

    # -- public API -----------------------------------------------------------

    def _rss_urls(self) -> List[str]:
        raw = settings.reddit_rss_feeds or ""
        urls = [u.strip() for u in raw.split(",") if u.strip()]
        # Safety: keep only reddit RSS URLs
        return [u for u in urls if "reddit.com/r/" in u and u.endswith("/.rss")]

    @staticmethod
    def _parse_rss_feed(xml_text: str, team_name: str, max_posts: int = 5) -> str:
        """Extract title/content from Atom RSS and filter by team tokens."""
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return ""

        ns = {"a": "http://www.w3.org/2005/Atom"}
        q = (team_name or "").lower().strip()
        tokens = [t for t in q.replace("-", " ").split() if len(t) >= 3]

        parts: List[str] = []
        kept = 0
        for e in root.findall("a:entry", ns):
            title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
            content = (e.findtext("a:content", default="", namespaces=ns) or "").strip()
            text = f"{title}\n{summary}\n{content}".lower()

            if q and q not in text and not any(tok in text for tok in tokens):
                continue

            if title:
                parts.append(title)
            if summary:
                parts.append(summary)
            elif content:
                parts.append(content)
            kept += 1
            if kept >= max_posts:
                break

        return "\n\n".join(parts)

    async def fetch_team_sentiment_posts(self, team_name: str, cache_ttl: int = 3600) -> str:
        """Fetch team-related Reddit text from configured subreddit RSS feeds.

        Uses curated feed URLs from ``REDDIT_RSS_FEEDS`` to ensure broad league
        coverage (incl. Champions League) with deterministic source scope.
        Falls back to JSON listing endpoints when RSS yields no hit.
        """
        cache_key = f"reddit:posts:{team_name.lower()}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return str(cached)

        session = self._ensure_session()

        # 1) RSS-first path (preferred)
        rss_urls = self._rss_urls()
        rss_parts: List[str] = []
        for url in rss_urls:
            try:
                await self._respect_rate_limit()
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    xml_text = await resp.text()
                    hit = self._parse_rss_feed(xml_text, team_name, max_posts=2)
                    if hit:
                        rss_parts.append(hit)
                    if len(rss_parts) >= 3:
                        break
            except Exception:
                continue

        if rss_parts:
            text = "\n\n".join(rss_parts)
            cache.set_json(cache_key, text, cache_ttl)
            return text

        # 2) Legacy JSON fallback
        await self._respect_rate_limit()
        token = await self._ensure_oauth_token()
        headers = {}
        listing_urls = []
        if token:
            headers["Authorization"] = f"Bearer {token}"
            listing_urls.extend([
                f"{_OAUTH_BASE_URL}/r/sports/new",
                f"{_OAUTH_BASE_URL}/r/hockey/new",
                f"{_OAUTH_BASE_URL}/r/soccer/new",
                f"{_OAUTH_BASE_URL}/r/tennis/new",
                f"{_OAUTH_BASE_URL}/r/nba/new",
                f"{_OAUTH_BASE_URL}/r/nfl/new",
            ])
        listing_urls.extend([
            f"{_BASE_URL}/r/sports/new.json",
            f"{_BASE_URL}/r/hockey/new.json",
            f"{_BASE_URL}/r/soccer/new.json",
            f"{_BASE_URL}/r/tennis/new.json",
            f"{_BASE_URL}/r/nba/new.json",
            f"{_BASE_URL}/r/nfl/new.json",
            f"{_OLD_BASE_URL}/r/sports/new.json",
        ])

        params = {"limit": "50", "raw_json": "1"}
        data = None
        try:
            for url in listing_urls:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json(content_type=None)
                    break
        except Exception:
            data = None

        if not data:
            return ""

        text = self._parse_posts(data, team_name)
        cache.set_json(cache_key, text, cache_ttl)
        return text

    # -- internal helpers -----------------------------------------------------

    @staticmethod
    def _parse_posts(data: dict, team_name: str) -> str:
        """Extract ``title`` + ``selftext`` from Reddit listing JSON.

        Uses local keyword filtering against `team_name` to replace the
        unreliable search endpoint in unauthenticated mode.
        """
        try:
            children: List[dict] = data.get("data", {}).get("children", [])
        except (AttributeError, TypeError):
            log.debug("reddit_parse: unexpected payload structure for '%s'", team_name)
            return ""

        q = (team_name or "").lower().strip()
        tokens = [t for t in q.replace("-", " ").split() if len(t) >= 3]

        parts: List[str] = []
        kept = 0
        for child in children:
            post = child.get("data", {})
            title = (post.get("title") or "").strip()
            body = (post.get("selftext") or "").strip()
            text = f"{title}\n{body}".lower()

            # Keep only posts mentioning the team query/tokens
            if q and q not in text and not any(tok in text for tok in tokens):
                continue

            if title:
                parts.append(title)
            if body:
                parts.append(body)
            kept += 1
            if kept >= 5:
                break

        return "\n\n".join(parts)

    # -- sync convenience wrapper ---------------------------------------------

    def fetch_team_sentiment_posts_sync(self, team_name: str, cache_ttl: int = 3600) -> str:
        """Synchronous wrapper — safe to call from inside a running event loop.

        Ensures the ``aiohttp`` session is closed within the same event
        loop that created it, preventing ``ResourceWarning`` leaks.
        Never raises to caller; returns empty string on errors.
        """
        async def _fetch_and_close() -> str:
            try:
                return await self.fetch_team_sentiment_posts(team_name, cache_ttl=cache_ttl)
            finally:
                await self.close()

        try:
            return _safe_sync_run(_fetch_and_close())
        except Exception as exc:
            log.warning("reddit_sync_wrapper_error: %s", exc)
            return ""
