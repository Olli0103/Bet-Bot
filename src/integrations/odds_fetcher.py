import asyncio
from typing import Dict, Any

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher


class OddsFetcher(AsyncBaseFetcher):
    def __init__(self):
        super().__init__(base_url=settings.odds_api_base_url)

    async def get_sports_async(self, ttl_seconds: int = 3600):
        cache_key = "odds:sports:list"
        cached = cache.get_json(cache_key)
        if cached:
            return cached
        data = await self.get("sports", params={"apiKey": settings.odds_api_key})
        cache.set_json(cache_key, data, ttl_seconds)
        return data

    async def get_sport_odds_async(self, sport_key: str, regions: str = "eu", markets: str = "h2h", ttl_seconds: int = 600):
        """Fetch odds with longer cache (10 min default) to save API calls."""
        cache_key = f"odds:{sport_key}:{regions}:{markets}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached
        data = await self.get(
            f"sports/{sport_key}/odds",
            params={
                "apiKey": settings.odds_api_key,
                "regions": regions,
                "markets": markets,
                "bookmakers": "tipico_de,pinnacle,bet365,betfair_ex_uk",
            },
        )
        cache.set_json(cache_key, data, ttl_seconds)
        return data

    async def get_historical_odds_async(self, sport_key: str, regions: str = "eu", markets: str = "h2h", days_history: int = 7):
        """Fetch historical odds for backtesting/training."""
        cache_key = f"odds:history:{sport_key}:{days_history}d:{regions}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached
        
        # Use the history endpoint if available, otherwise fetch current and store
        try:
            data = await self.get(
                f"sports/{sport_key}/odds/history",
                params={
                    "apiKey": settings.odds_api_key,
                    "regions": regions,
                    "markets": markets,
                    "daysFromNow": days_history,
                    "bookmakers": "tipico_de,pinnacle,bet365,betfair_ex_uk",
                },
            )
            # Cache historical data longer (24 hours)
            cache.set_json(cache_key, data, ttl_seconds=86400)
            return data
        except Exception as e:
            print(f"Historical odds not available: {e}")
            return None

    # sync helpers for non-async call sites
    def get_sports(self, ttl_seconds: int = 3600):
        return asyncio.run(self.get_sports_async(ttl_seconds=ttl_seconds))

    def get_sport_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h", ttl_seconds: int = 600):
        return asyncio.run(self.get_sport_odds_async(sport_key=sport_key, regions=regions, markets=markets, ttl_seconds=ttl_seconds))

    def get_historical_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h", days_history: int = 7):
        return asyncio.run(self.get_historical_odds_async(sport_key=sport_key, regions=regions, markets=markets, days_history=days_history))
