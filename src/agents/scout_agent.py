"""Scout Agent: Continuously monitors odds for steam moves and Twitter for breaking news.

The Scout is the first agent in the pipeline. It detects anomalous price
movements and breaking injury news, then triggers the Analyst for deeper
investigation when something interesting is found.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.core.settings import settings
from src.core.volatility_tracker import (
    detect_steam_move,
    get_volatility,
    record_odds_snapshot,
)
from src.data.redis_cache import cache
from src.integrations.odds_fetcher import OddsFetcher

log = logging.getLogger(__name__)

SNAPSHOT_KEY = "scout:odds_snapshot"
ALERTS_KEY = "scout:alerts"
KICKOFF_TIMES_KEY = "orchestrator:kickoff_times"


class ScoutAgent:
    """Monitors OddsAPI for steam moves (rapid price changes) and Twitter for injuries."""

    def __init__(self) -> None:
        self.odds_fetcher = OddsFetcher()
        self._previous_snapshot: Dict[str, Dict[str, float]] = {}

    async def monitor_odds(self) -> List[Dict[str, Any]]:
        """Poll odds and detect steam moves. Returns list of alert dicts."""
        alerts: List[Dict[str, Any]] = []

        # Load previous snapshot from Redis
        prev = cache.get_json(SNAPSHOT_KEY) or {}
        self._previous_snapshot = prev

        sports = [s.strip() for s in settings.live_sports.split(",") if s.strip()]
        current_snapshot: Dict[str, Dict[str, float]] = {}
        kickoff_times: List[str] = []

        for sport in sports:
            try:
                events = await self.odds_fetcher.get_sport_odds_async(
                    sport_key=sport, regions="eu", markets="h2h", ttl_seconds=60,
                )
            except Exception:
                continue
            if not isinstance(events, list):
                continue

            for event in events:
                event_id = str(event.get("id") or "")
                home = event.get("home_team") or ""
                away = event.get("away_team") or ""
                # Collect kickoff times for adaptive polling
                commence = event.get("commence_time")
                if commence:
                    kickoff_times.append(str(commence))
                if not event_id or not home:
                    continue

                # Extract current odds from sharp books
                for bm in event.get("bookmakers", []):
                    if bm.get("key") not in ("pinnacle", "betfair_ex_uk"):
                        continue
                    for mkt in bm.get("markets", []):
                        if mkt.get("key") != "h2h":
                            continue
                        for outcome in mkt.get("outcomes", []):
                            name = outcome.get("name")
                            price = outcome.get("price")
                            if not name or not price:
                                continue

                            key = f"{event_id}:{name}"
                            current_odds = float(price)
                            current_ip = 1.0 / current_odds if current_odds > 1.0 else 0.0
                            current_snapshot[key] = {"odds": current_odds, "ip": current_ip}

                            # Record for volatility tracking
                            try:
                                record_odds_snapshot(name, current_ip)
                            except Exception:
                                pass

                            # Check for steam move
                            prev_data = prev.get(key, {})
                            prev_odds = prev_data.get("odds", current_odds)

                            team_vol = get_volatility(name)
                            is_steam = detect_steam_move(current_odds, prev_odds, team_vol)

                            if is_steam:
                                alerts.append({
                                    "type": "steam_move",
                                    "event_id": event_id,
                                    "sport": sport,
                                    "home": home,
                                    "away": away,
                                    "selection": name,
                                    "prev_odds": prev_odds,
                                    "current_odds": current_odds,
                                    "movement_pct": round(abs(1.0 / current_odds - 1.0 / prev_odds) * 100, 2),
                                    "team_volatility": team_vol,
                                })
                                log.info("Steam move detected: %s %s %.2f -> %.2f", sport, name, prev_odds, current_odds)

        # Save current snapshot for next comparison
        cache.set_json(SNAPSHOT_KEY, current_snapshot, ttl_seconds=3600)

        # Cache kickoff times so the orchestrator can use adaptive polling
        if kickoff_times:
            cache.set_json(KICKOFF_TIMES_KEY, kickoff_times, ttl_seconds=3600)

        # Save alerts
        if alerts:
            existing_alerts = cache.get_json(ALERTS_KEY) or []
            existing_alerts.extend(alerts)
            cache.set_json(ALERTS_KEY, existing_alerts[-50:], ttl_seconds=6 * 3600)

        return alerts

    async def monitor_twitter(self) -> List[Dict[str, Any]]:
        """Monitor Twitter/X for breaking injury news."""
        alerts: List[Dict[str, Any]] = []

        if not settings.twitter_enabled:
            return alerts

        try:
            from src.integrations.twitter_fetcher import TwitterFetcher
            tw = TwitterFetcher()
        except ImportError:
            return alerts

        # Get teams from currently tracked events
        sports = [s.strip() for s in settings.live_sports.split(",") if s.strip()]
        teams: set = set()

        for sport in sports:
            try:
                events = await self.odds_fetcher.get_sport_odds_async(
                    sport_key=sport, regions="eu", markets="h2h", ttl_seconds=300,
                )
                if isinstance(events, list):
                    for e in events:
                        if e.get("home_team"):
                            teams.add(e["home_team"])
                        if e.get("away_team"):
                            teams.add(e["away_team"])
            except Exception:
                continue

        for team in list(teams)[:20]:
            try:
                breaking = tw.check_breaking_injury(team)
                if breaking:
                    alerts.append({
                        "type": "breaking_injury",
                        "team": team,
                        "text": breaking,
                    })
                    log.info("Breaking injury detected: %s - %s", team, breaking[:80])
            except Exception:
                pass

        return alerts

    @staticmethod
    def get_recent_alerts() -> List[Dict[str, Any]]:
        """Retrieve recent scout alerts from Redis."""
        return cache.get_json(ALERTS_KEY) or []
