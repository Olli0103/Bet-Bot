import asyncio
from typing import Dict, List, Any

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher


class APISportsFetcher(AsyncBaseFetcher):
    def __init__(self):
        headers = {
            "x-apisports-key": settings.apisports_api_key,
            "Accept": "application/json",
        }
        super().__init__(base_url=settings.apisports_base_url, headers=headers)

    async def get_fixtures_by_ids_async(self, fixture_ids: List[int], ttl_seconds: int = 120) -> Dict[str, Any]:
        ids_csv = "-".join(str(i) for i in fixture_ids)
        cache_key = f"apisports:fixtures:ids:{ids_csv}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached
        data = await self.get("fixtures", params={"ids": ids_csv})
        cache.set_json(cache_key, data, ttl_seconds)
        return data

    async def get_injuries_by_fixture_ids_async(self, fixture_ids: List[int], ttl_seconds: int = 180) -> Dict[str, Any]:
        ids_csv = "-".join(str(i) for i in fixture_ids)
        cache_key = f"apisports:injuries:ids:{ids_csv}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached
        data = await self.get("injuries", params={"ids": ids_csv})
        cache.set_json(cache_key, data, ttl_seconds)
        return data

    def get_fixtures_by_ids(self, fixture_ids: List[int], ttl_seconds: int = 120) -> Dict[str, Any]:
        return asyncio.run(self.get_fixtures_by_ids_async(fixture_ids, ttl_seconds))

    def get_injuries_by_fixture_ids(self, fixture_ids: List[int], ttl_seconds: int = 180) -> Dict[str, Any]:
        return asyncio.run(self.get_injuries_by_fixture_ids_async(fixture_ids, ttl_seconds))
