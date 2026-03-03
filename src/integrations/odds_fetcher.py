import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)


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

    async def get_sport_odds_async(
        self,
        sport_key: str,
        regions: str = "eu",
        markets: str = "h2h,spreads,totals",
        ttl_seconds: int = 600,
    ):
        """Fetch odds for multiple markets with longer cache (10 min default)."""
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
                "bookmakers": "tipico_de,pinnacle,bet365,betfair_ex_uk,betsson,unibet",
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
                    "bookmakers": "tipico_de,pinnacle,bet365,betfair_ex_uk,betsson,unibet",
                },
            )
            cache.set_json(cache_key, data, ttl_seconds=86400)
            return data
        except Exception as e:
            print(f"Historical odds not available: {e}")
            return None

    async def get_odds_12h_ago_async(
        self, sport_key: str, regions: str = "eu", markets: str = "h2h"
    ) -> Dict[str, Dict[str, float]]:
        """Fetch historical odds from ~12 hours ago for momentum calculation.

        Uses the Pro-Tier /history endpoint with a specific date parameter.
        Returns {event_id: {selection: implied_probability_12h_ago}}.
        """
        cache_key = f"odds:momentum:{sport_key}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached

        # Calculate ISO timestamp for 12 hours ago
        ts_12h_ago = datetime.now(timezone.utc) - timedelta(hours=12)
        date_str = ts_12h_ago.strftime("%Y-%m-%dT%H:%M:%SZ")

        result: Dict[str, Dict[str, float]] = {}
        try:
            data = await self.get(
                f"sports/{sport_key}/odds-history",
                params={
                    "apiKey": settings.odds_api_key,
                    "regions": regions,
                    "markets": markets,
                    "date": date_str,
                    "bookmakers": "pinnacle,betfair_ex_uk,bet365",
                },
            )
            if isinstance(data, dict):
                # The history endpoint wraps events in a "data" key
                events = data.get("data", [])
            elif isinstance(data, list):
                events = data
            else:
                events = []

            for event in events:
                event_id = str(event.get("id") or "")
                if not event_id:
                    continue
                for bm in event.get("bookmakers", []):
                    if bm.get("key") not in ("pinnacle", "betfair_ex_uk", "bet365"):
                        continue
                    for mkt in bm.get("markets", []):
                        if mkt.get("key") != markets.split(",")[0]:
                            continue
                        for outcome in mkt.get("outcomes", []):
                            name = outcome.get("name")
                            price = outcome.get("price")
                            if name and price and float(price) > 1.0:
                                if event_id not in result:
                                    result[event_id] = {}
                                result[event_id][name] = round(1.0 / float(price), 4)
                    break  # Take first sharp book found

            cache.set_json(cache_key, result, ttl_seconds=3600)
        except Exception as exc:
            log.debug("Historical odds 12h ago not available for %s: %s", sport_key, exc)

        return result

    # sync helpers for non-async call sites (loop-safe)
    def get_sports(self, ttl_seconds: int = 3600):
        return _safe_sync_run(self.get_sports_async(ttl_seconds=ttl_seconds))

    def get_sport_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h,spreads,totals", ttl_seconds: int = 600):
        return _safe_sync_run(self.get_sport_odds_async(sport_key=sport_key, regions=regions, markets=markets, ttl_seconds=ttl_seconds))

    def get_historical_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h", days_history: int = 7):
        return _safe_sync_run(self.get_historical_odds_async(sport_key=sport_key, regions=regions, markets=markets, days_history=days_history))

    def get_odds_12h_ago(self, sport_key: str, regions: str = "eu", markets: str = "h2h") -> Dict[str, Dict[str, float]]:
        return _safe_sync_run(self.get_odds_12h_ago_async(sport_key=sport_key, regions=regions, markets=markets))
