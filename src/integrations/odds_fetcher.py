import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import httpx

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import APIFetchError, AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)

# How long to cache the /v4/sports list (avoids redundant API calls)
_ACTIVE_SPORTS_TTL = 2 * 3600  # 2 hours

# Deep-Sleep: sports with zero events in today's trading window
_DEEP_SLEEP_KEY = "odds:deep_sleep_sports"
_DEEP_SLEEP_TTL = 24 * 3600  # 24 hours — re-check once per day

# Global rate limiter — protects the API key across all workers
_RATE_LIMIT_KEY = "odds:api:rate_limit"
_MIN_REQUEST_INTERVAL_S = 1.0  # minimum 1s between requests to the Odds API


def _check_global_rate_limit() -> bool:
    """Check whether enough time has passed since the last Odds API request.

    Uses a Redis key with the last-request timestamp.  If the interval
    since the last call is smaller than ``_MIN_REQUEST_INTERVAL_S``, the
    caller should skip or wait.  This prevents a manual "Odds Refresh"
    button from triggering a redundant external API call if the data is
    still fresh (< interval).

    Returns True if OK to proceed, False if rate-limited.
    """
    raw = cache.get(_RATE_LIMIT_KEY)
    if raw:
        try:
            last_ts = float(raw)
            if time.time() - last_ts < _MIN_REQUEST_INTERVAL_S:
                return False
        except (ValueError, TypeError):
            pass
    return True


def _record_api_call() -> None:
    """Record the current timestamp in Redis for rate-limit tracking."""
    cache.set(_RATE_LIMIT_KEY, str(time.time()), ttl_seconds=60)


def get_trading_window_end() -> datetime:
    """Return the end of the current trading session (tomorrow 06:59 UTC).

    The bot scopes all API requests to "now → tomorrow 06:59 UTC".
    Events beyond this window are illiquid noise with no sharp-money
    signal — polling them wastes API credits without generating edge.
    """
    now = datetime.now(timezone.utc)
    # Next day at 07:00 UTC is the session boundary
    tomorrow_7am = (now + timedelta(days=1)).replace(
        hour=6, minute=59, second=59, microsecond=0
    )
    return tomorrow_7am


def get_markets_for_sport(sport_key: str) -> str:
    """Return the markets parameter based on league liquidity.

    Illiquid leagues (Tier 3/4) only get h2h — requesting spreads/totals
    for niche markets wastes API credits on data with no sharp-book pricing.
    """
    s = sport_key.lower()

    # Check most specific (semi-liquid) before broader (liquid) to avoid
    # "bundesliga" matching before "bundesliga2"
    _SEMI_LIQUID = (
        "soccer_germany_bundesliga2", "soccer_efl_champ",
        "soccer_usa_mls", "soccer_england_league1",
        "soccer_england_league2",
    )
    for key in _SEMI_LIQUID:
        if key in s:
            return "h2h,totals"

    # Tier 1+2: full market coverage
    _LIQUID = (
        "soccer_epl", "soccer_spain_la_liga", "soccer_uefa_champs_league",
        "americanfootball_nfl", "basketball_nba", "soccer_uefa_europa_league",
        "soccer_germany_bundesliga", "soccer_italy_serie_a",
        "soccer_france_ligue_one", "icehockey_nhl", "basketball_euroleague",
        "soccer_brazil_serie_a", "soccer_portugal_primeira_liga",
        "soccer_netherlands_eredivisie",
    )
    for key in _LIQUID:
        if key in s:
            return "h2h,spreads,totals"

    # Tier 4 / unknown: h2h only
    return "h2h"


class OddsFetcher(AsyncBaseFetcher):
    def __init__(self):
        # Higher timeout for heavy odds endpoints (h2h+spreads+totals)
        # to avoid false TimeoutError trips on busy slates.
        super().__init__(base_url=settings.odds_api_base_url, timeout=60)

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
        except (httpx.HTTPError, APIFetchError, asyncio.TimeoutError) as exc:
            log.warning("Failed to fetch active sports from API (%s): %s",
                        type(exc).__name__, exc)
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
        commence_time_from: Optional[str] = None,
        commence_time_to: Optional[str] = None,
    ):
        """Fetch odds scoped to the active trading window.

        Parameters
        ----------
        commence_time_from, commence_time_to : str, optional
            ISO-8601 timestamps to scope the API response to a time window.
            When provided, only events within this window are returned,
            saving API credits by excluding far-future illiquid markets.
        """
        cache_key = f"odds:{sport_key}:{regions}:{markets}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached

        # Global rate limiter: skip redundant calls across all workers
        if not _check_global_rate_limit():
            log.debug("Rate-limited: skipping redundant API call for %s", sport_key)
            return cached or []

        params = {
            "apiKey": settings.odds_api_key,
            "regions": regions,
            "markets": markets,
            "bookmakers": "tipico_de,pinnacle,bet365,betfair_ex_uk,betsson,unibet",
        }
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to

        try:
            _record_api_call()
            data = await self.get(
                f"sports/{sport_key}/odds",
                params=params,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            # Plan/entitlement fallback: retry with minimal request shape.
            # Some keys allow /sports but reject premium bookmakers/markets.
            if status in (401, 422):
                log.warning(
                    "Odds API %s for %s with full params; retrying minimal mode",
                    status,
                    sport_key,
                )
                fallback_params = {
                    "apiKey": settings.odds_api_key,
                    "regions": regions,
                    "markets": "h2h",
                }
                if commence_time_from:
                    fallback_params["commenceTimeFrom"] = commence_time_from
                if commence_time_to:
                    fallback_params["commenceTimeTo"] = commence_time_to
                _record_api_call()
                data = await self.get(
                    f"sports/{sport_key}/odds",
                    params=fallback_params,
                )
            else:
                raise

        # Empty Window Cache: if the API returns an empty list ([]),
        # cache it until the trading window ends to prevent the fetcher
        # from hammering the API in an endless loop for cold sports.
        if isinstance(data, list) and len(data) == 0:
            window_end = get_trading_window_end()
            now = datetime.now(timezone.utc)
            ttl_to_window_end = max(3600, int((window_end - now).total_seconds()))
            cache.set_json(cache_key, data, ttl_seconds=ttl_to_window_end)
            log.info("Empty window cached for %s (TTL=%ds until next session)", sport_key, ttl_to_window_end)
        else:
            cache.set_json(cache_key, data, ttl_seconds)

        return data

    def is_sport_deep_sleeping(self, sport_key: str) -> bool:
        """Check if a sport is in deep-sleep (no events in today's window)."""
        sleeping = cache.get_json(_DEEP_SLEEP_KEY) or []
        return sport_key in sleeping

    def mark_sport_deep_sleep(self, sport_key: str) -> None:
        """Put a sport into deep-sleep for 24 hours (no events today)."""
        sleeping = cache.get_json(_DEEP_SLEEP_KEY) or []
        if sport_key not in sleeping:
            sleeping.append(sport_key)
            cache.set_json(_DEEP_SLEEP_KEY, sleeping, ttl_seconds=_DEEP_SLEEP_TTL)
            log.info("Deep-sleep activated for %s (no events in trading window)", sport_key)

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
        except (httpx.HTTPError, APIFetchError, asyncio.TimeoutError) as exc:
            log.debug("Historical odds not available for %s (%s): %s",
                       sport_key, type(exc).__name__, exc)
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
        except (httpx.HTTPError, APIFetchError, asyncio.TimeoutError) as exc:
            log.debug("Historical odds 12h ago not available for %s (%s): %s",
                       sport_key, type(exc).__name__, exc)

        return result

    async def get_event_odds(self, event_id: str) -> Optional[Dict[str, float]]:
        """Fetch current odds for a single event by ID.

        Returns ``{selection_name: decimal_odds}`` or None.  Uses a short
        TTL cache (10s) to allow rapid re-checks from the tip publisher
        without burning API quota.
        """
        cache_key = f"odds:event:{event_id}"
        cached = cache.get_json(cache_key)
        if cached:
            return cached

        if not _check_global_rate_limit():
            return None

        try:
            _record_api_call()
            data = await self.get(
                f"sports/upcoming/odds",
                params={
                    "apiKey": settings.odds_api_key,
                    "regions": "eu",
                    "markets": "h2h",
                    "eventIds": event_id,
                    "bookmakers": "tipico_de,pinnacle",
                },
            )
            result: Dict[str, float] = {}
            if isinstance(data, list):
                for event in data:
                    if str(event.get("id", "")) == event_id:
                        for bm in event.get("bookmakers", []):
                            for mkt in bm.get("markets", []):
                                for outcome in mkt.get("outcomes", []):
                                    name = outcome.get("name")
                                    price = outcome.get("price")
                                    if name and price and float(price) > 1.0:
                                        result[name] = float(price)
                            if result:
                                break
            if result:
                cache.set_json(cache_key, result, ttl_seconds=10)
            return result or None
        except (httpx.HTTPError, APIFetchError, asyncio.TimeoutError) as exc:
            log.warning("get_event_odds failed for %s (%s): %s",
                        event_id, type(exc).__name__, exc)
            return None

    # sync helpers for non-async call sites (loop-safe)
    def get_sports(self, ttl_seconds: int = 3600):
        return _safe_sync_run(self.get_sports_async(ttl_seconds=ttl_seconds))

    def get_sport_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h,spreads,totals", ttl_seconds: int = 600):
        return _safe_sync_run(
            self.get_sport_odds_async(
                sport_key=sport_key,
                regions=regions,
                markets=markets,
                ttl_seconds=ttl_seconds,
            ),
            timeout=90,
        )

    def get_historical_odds(self, sport_key: str, regions: str = "eu", markets: str = "h2h", days_history: int = 7):
        return _safe_sync_run(self.get_historical_odds_async(sport_key=sport_key, regions=regions, markets=markets, days_history=days_history))

    def get_odds_12h_ago(self, sport_key: str, regions: str = "eu", markets: str = "h2h") -> Dict[str, Dict[str, float]]:
        return _safe_sync_run(self.get_odds_12h_ago_async(sport_key=sport_key, regions=regions, markets=markets))
