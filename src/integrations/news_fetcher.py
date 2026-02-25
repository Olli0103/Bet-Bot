import asyncio
from typing import Dict, Any

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher


class NewsFetcher(AsyncBaseFetcher):
    def __init__(self):
        super().__init__(base_url=settings.newsapi_base_url)

    async def search_news_async(self, query: str, language: str = "en", ttl_seconds: int = 600) -> Dict[str, Any]:
        cache_key = f"news:{query}:{language}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached
        data = await self.get(
            "everything",
            params={
                "q": query,
                "language": language,
                "sortBy": "publishedAt",
                "apiKey": settings.newsapi_key,
                "pageSize": 50,
            },
        )
        cache.set_json(cache_key, data, ttl_seconds)
        return data

    def search_news(self, query: str, language: str = "en", ttl_seconds: int = 600) -> Dict[str, Any]:
        return asyncio.run(self.search_news_async(query=query, language=language, ttl_seconds=ttl_seconds))
