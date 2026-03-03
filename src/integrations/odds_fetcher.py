import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)

# How long to cache the /v4/sports list (avoids redundant API calls)
_ACTIVE_SPORTS_TTL = 2 * 3600  # 2 hours


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

    async def get_active_sports_from_api_async(self) -> List[str]:
        """Fetch the list of currently in-season sport keys from the Odds API.

        Calls ``/v4/sports`` and caches the result in Redis for 2 hours.
        Returns a list of active sport keys like ``["tennis_atp_dubai", ...]``.
        """
        cache_key = "odds:active_sport_keys"
        cached = cache.get_json(cache_key)
        if cached and isinstance(cached, list):
            return cached

        try:
            data = await self.get("sports", params={"apiKey": settings.odds_api_key})
        except Exception as exc:
            log.warning("Failed to fetch active sports from API: %s", exc)
            return []

        if not isinstance(data, list):
            return []

        keys = [
            str(s["key"])
            for s in data
            if isinstance(s, dict) and s.get("key") and s.get("active")
        ]
        cache.set_json(cache_key, keys, ttl_seconds=_ACTIVE_SPORTS_TTL)
        log.debug("Cached %d active sport keys from Odds API", len(keys))
        return keys

    def get_active_sports_from_api(self) -> List[str]:
        """Sync wrapper for :meth:`get_active_sports_from_api_async`."""
        return _safe_sync_run(self.get_active_sports_from_api_async())

    @staticmethod
    def resolve_sport_keys(
        user_base_keys: List[str],
        api_active_keys: List[str],
    ) -> List[str]:
        """Expand user base-keys into exact API keys via prefix matching.

        A user setting of ``tennis_atp`` will match ``tennis_atp_dubai``,
        ``tennis_atp_wimbledon``, ``tennis_atp_challenger``, etc.
        If a base key appears verbatim in the active list, it is included as-is.

        Falls back to the original base key when the active list is empty
        (API unavailable) so the bot doesn't go silent.
        """
        if not api_active_keys:
            return list(user_base_keys)

        active_set = set(api_active_keys)
        resolved: List[str] = []
        seen: set = set()

        for base in user_base_keys:
            if base in active_set:
                if base not in seen:
                    resolved.append(base)
                    seen.add(base)
            # Also find sub-keys that start with base + "_"
            for ak in sorted(api_active_keys):
                if ak.startswith(base + "_") or ak == base:
                    if ak not in seen:
                        resolved.append(ak)
                        seen.add(ak)
            # Fallback: if nothing matched, keep the base key anyway
            if base not in seen and not any(ak.startswith(base + "_") for ak in api_active_keys):
                resolved.append(base)
                seen.add(base)

        return resolved

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
