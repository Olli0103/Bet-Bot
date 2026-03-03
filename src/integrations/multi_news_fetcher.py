"""Multi-source news fetcher with cascading fallback, dedup, and rate limiting.

Provider priority:
  1. Rotowire RSS  (free, sports-focused, primary)
  2. GNews API     (fallback 1, requires GNEWS_API_KEY)
  3. NewsData.io   (fallback 2, requires NEWSDATA_API_KEY)
  4. NewsAPI       (fallback 3, requires NEWSAPI_KEY)

Features:
  - Per-source request budget with rate-limit guard
  - Redis caching (TTL 30–120 min) to reduce quota burn
  - Dedup by URL/title hash across sources
  - Source confidence scoring in metadata
  - Graceful degradation (no pipeline crash when one source fails)
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)

# Source confidence scores (higher = more trusted for sports)
SOURCE_CONFIDENCE: Dict[str, float] = {
    "rotowire": 0.95,
    "gnews": 0.70,
    "newsdata": 0.65,
    "newsapi": 0.75,
}

# Per-source rate limits (requests per hour)
_SOURCE_BUDGET: Dict[str, int] = {
    "rotowire": 120,   # RSS, generous
    "gnews": 100,       # free tier: 100/day, be conservative
    "newsdata": 200,    # free tier: 200/day
    "newsapi": 100,     # free tier: 100/day
}

_CACHE_TTL_SECONDS = 60 * 60  # 60 min default


def _title_hash(title: str) -> str:
    """Hash a title for dedup purposes (case-insensitive, stripped)."""
    normalized = (title or "").strip().lower()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]


def _url_hash(url: str) -> str:
    """Hash a URL for dedup purposes."""
    return hashlib.md5((url or "").strip().encode("utf-8")).hexdigest()[:16]


def _get_ratelimit_key(source: str) -> str:
    """Generate an hourly-bucketed rate-limit key.

    Returns e.g. ``"news_ratelimit:gnews:2026-03-03_18"`` so each hour
    gets its own counter.  Even if ``cache.set()`` resets the TTL, the
    key naturally rotates every hour — no "leaky TTL" lockout.
    """
    current_hour = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H")
    return f"news_ratelimit:{source}:{current_hour}"


def _check_rate_limit(source: str) -> bool:
    """Check if source is within its hourly request budget."""
    key = _get_ratelimit_key(source)
    count_str = cache.get(key)
    if count_str is None:
        return True
    try:
        return int(count_str) < _SOURCE_BUDGET.get(source, 100)
    except (ValueError, TypeError):
        return True


def _increment_rate_limit(source: str) -> None:
    """Increment the hourly request counter for a source."""
    key = _get_ratelimit_key(source)
    count_str = cache.get(key)
    try:
        count = int(count_str) + 1 if count_str else 1
    except (ValueError, TypeError):
        count = 1
    # TTL still set so old hour-buckets get cleaned up automatically
    cache.set(key, str(count), ttl_seconds=7200)


class GNewsFetcher(AsyncBaseFetcher):
    """GNews API client (https://gnews.io)."""

    def __init__(self):
        super().__init__(base_url=settings.gnews_base_url)

    async def search_async(self, query: str, language: str = "en", max_results: int = 10) -> List[Dict[str, Any]]:
        if not settings.gnews_api_key:
            return []

        cache_key = f"gnews:{_title_hash(query)}:{language}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        if not _check_rate_limit("gnews"):
            log.debug("GNews rate limit reached, skipping")
            return []

        try:
            data = await self.get("search", params={
                "q": query,
                "lang": language,
                "max": max_results,
                "apikey": settings.gnews_api_key,
            })
            _increment_rate_limit("gnews")
            articles = data.get("articles") or []
            results = [
                {
                    "title": a.get("title", ""),
                    "summary": (a.get("description") or "")[:500],
                    "url": a.get("url", ""),
                    "published": a.get("publishedAt", ""),
                    "source": "gnews",
                    "source_name": (a.get("source") or {}).get("name", ""),
                    "confidence": SOURCE_CONFIDENCE["gnews"],
                }
                for a in articles
            ]
            cache.set_json(cache_key, results, _CACHE_TTL_SECONDS)
            return results
        except Exception as exc:
            log.warning("GNews search failed for '%s': %s", query, exc)
            return []


class NewsDataFetcher(AsyncBaseFetcher):
    """NewsData.io API client."""

    def __init__(self):
        super().__init__(base_url=settings.newsdata_base_url)

    async def search_async(self, query: str, language: str = "en", max_results: int = 10) -> List[Dict[str, Any]]:
        if not settings.newsdata_api_key:
            return []

        cache_key = f"newsdata:{_title_hash(query)}:{language}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        if not _check_rate_limit("newsdata"):
            log.debug("NewsData rate limit reached, skipping")
            return []

        try:
            data = await self.get("news", params={
                "q": query,
                "language": language,
                "apikey": settings.newsdata_api_key,
            })
            _increment_rate_limit("newsdata")
            articles = data.get("results") or []
            results = [
                {
                    "title": a.get("title", ""),
                    "summary": (a.get("description") or "")[:500],
                    "url": a.get("link", ""),
                    "published": a.get("pubDate", ""),
                    "source": "newsdata",
                    "source_name": a.get("source_id", ""),
                    "confidence": SOURCE_CONFIDENCE["newsdata"],
                }
                for a in articles[:max_results]
            ]
            cache.set_json(cache_key, results, _CACHE_TTL_SECONDS)
            return results
        except Exception as exc:
            log.warning("NewsData search failed for '%s': %s", query, exc)
            return []


class MultiNewsFetcher:
    """Cascading multi-source news fetcher with dedup and rate limiting.

    Tries sources in priority order. Stops early when enough articles are
    collected. Deduplicates by URL and title hash across all sources.
    """

    def __init__(self, min_articles: int = 5, max_articles: int = 20):
        self.min_articles = min_articles
        self.max_articles = max_articles

    def _dedup_add(
        self,
        articles: List[Dict[str, Any]],
        seen: set,
        entry: Dict[str, Any],
    ) -> None:
        """Add entry to articles list if not a duplicate (by title/URL hash)."""
        th = _title_hash(entry.get("title", ""))
        url = entry.get("url", "")
        uh = _url_hash(url) if url else None

        if th in seen:
            return
        if uh and uh in seen:
            return

        seen.add(th)
        if uh:
            seen.add(uh)
        articles.append(entry)

    async def search_async(
        self,
        query: str,
        language: str = "en",
    ) -> List[Dict[str, Any]]:
        """Search across all sources with fallback cascade."""
        cache_key = f"multinews:{_title_hash(query)}:{language}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        all_articles: List[Dict[str, Any]] = []
        seen_hashes: set = set()

        # 1. Rotowire RSS (injury/sports news) — search across all feeds
        try:
            from src.integrations.rss_fetcher import RSSFetcher, SPORT_TO_FEED
            rss = RSSFetcher()
            # Use OddsAPI sport keys so _sport_key_to_feed can resolve them
            all_sport_keys = list(SPORT_TO_FEED.keys()) + ["soccer_epl"]
            rss_results = rss.fetch_all_sports_news(
                active_sports=all_sport_keys,
                team_names=[query],
                max_age_hours=48,
            )
            for entry in rss_results:
                self._dedup_add(all_articles, seen_hashes, {
                    **entry,
                    "source": "rotowire",
                    "confidence": SOURCE_CONFIDENCE["rotowire"],
                })
        except Exception as exc:
            log.debug("Rotowire RSS failed for '%s': %s", query, exc)

        # 2. GNews (if we need more articles)
        if len(all_articles) < self.min_articles:
            gnews = GNewsFetcher()
            try:
                gnews_results = await gnews.search_async(query, language)
                for a in gnews_results:
                    self._dedup_add(all_articles, seen_hashes, a)
            except Exception as exc:
                log.debug("GNews failed for '%s': %s", query, exc)
            finally:
                await gnews.close()

        # 3. NewsData.io (if still not enough)
        if len(all_articles) < self.min_articles:
            newsdata = NewsDataFetcher()
            try:
                newsdata_results = await newsdata.search_async(query, language)
                for a in newsdata_results:
                    self._dedup_add(all_articles, seen_hashes, a)
            except Exception as exc:
                log.debug("NewsData failed for '%s': %s", query, exc)
            finally:
                await newsdata.close()

        # 4. NewsAPI (final fallback)
        if len(all_articles) < self.min_articles:
            from src.integrations.news_fetcher import NewsFetcher
            nf = NewsFetcher()
            try:
                payload = await nf.search_news_async(query, language)
                for a in (payload.get("articles") or []):
                    self._dedup_add(all_articles, seen_hashes, {
                        "title": a.get("title", ""),
                        "summary": (a.get("description") or "")[:500],
                        "url": a.get("url", ""),
                        "published": a.get("publishedAt", ""),
                        "source": "newsapi",
                        "source_name": (a.get("source") or {}).get("name", ""),
                        "confidence": SOURCE_CONFIDENCE["newsapi"],
                    })
            except Exception as exc:
                log.debug("NewsAPI failed for '%s': %s", query, exc)
            finally:
                await nf.close()

        # Trim to max
        all_articles = all_articles[:self.max_articles]

        # Sort by confidence (highest first)
        all_articles.sort(key=lambda x: float(x.get("confidence", 0)), reverse=True)

        # Cache the merged result
        cache.set_json(cache_key, all_articles, _CACHE_TTL_SECONDS)

        log.info(
            "MultiNews: query='%s' returned %d articles (sources: %s)",
            query,
            len(all_articles),
            ", ".join(sorted({a.get("source", "?") for a in all_articles})),
        )

        return all_articles

    def search(self, query: str, language: str = "en") -> List[Dict[str, Any]]:
        """Sync wrapper for search_async."""
        return _safe_sync_run(self.search_async(query, language))

    def get_source_health(self) -> Dict[str, Dict[str, Any]]:
        """Return rate limit status for each source (current hour bucket)."""
        health: Dict[str, Dict[str, Any]] = {}
        for source, budget in _SOURCE_BUDGET.items():
            key = _get_ratelimit_key(source)
            count_str = cache.get(key)
            used = int(count_str) if count_str else 0
            health[source] = {
                "used": used,
                "budget": budget,
                "available": budget - used,
                "confidence": SOURCE_CONFIDENCE.get(source, 0),
            }
        return health
