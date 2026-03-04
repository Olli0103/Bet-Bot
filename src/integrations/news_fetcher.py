import logging
import re
from typing import Dict, Any

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)

# Characters that break GNews/NewsAPI query syntax
_QUERY_UNSAFE = re.compile(r"[^\w\s\-.'&]", re.UNICODE)
_MAX_QUERY_LEN = 100


def sanitize_news_query(raw_query: str) -> str:
    """Sanitize a team/match name for safe use as a news API query.

    - Strips special characters that cause 400 errors
    - Truncates to MAX_QUERY_LEN
    - Returns a fallback if result is empty
    """
    q = _QUERY_UNSAFE.sub(" ", raw_query).strip()
    # Collapse multiple spaces
    q = re.sub(r"\s+", " ", q)
    # Truncate
    if len(q) > _MAX_QUERY_LEN:
        q = q[:_MAX_QUERY_LEN].rsplit(" ", 1)[0]
    return q or "sports"


class NewsFetcher(AsyncBaseFetcher):
    def __init__(self):
        super().__init__(base_url=settings.newsapi_base_url)

    async def search_news_async(self, query: str, language: str = "en", ttl_seconds: int = 600) -> Dict[str, Any]:
        sanitized = sanitize_news_query(query)
        cache_key = f"news:{sanitized}:{language}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached
        try:
            data = await self.get(
                "everything",
                params={
                    "q": sanitized,
                    "language": language,
                    "sortBy": "publishedAt",
                    "apiKey": settings.newsapi_key,
                    "pageSize": 50,
                },
            )
            cache.set_json(cache_key, data, ttl_seconds)
            return data
        except Exception as exc:
            log.warning("news_fetch_failed: query=%s sanitized=%s error=%s",
                        query[:50], sanitized[:50], str(exc)[:100])
            # Return empty result instead of propagating
            empty = {"articles": [], "totalResults": 0, "status": "error"}
            cache.set_json(cache_key, empty, ttl_seconds=300)
            return empty

    def search_news(self, query: str, language: str = "en", ttl_seconds: int = 600) -> Dict[str, Any]:
        return _safe_sync_run(self.search_news_async(query=query, language=language, ttl_seconds=ttl_seconds))
