"""Scout Agent: Monitors odds for steam moves and aggregates injury intel.

The Scout is the first agent in the pipeline. It detects anomalous price
movements and breaking injury news (via API-Sports + Rotowire RSS),
then triggers the Analyst for deeper investigation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.core.dynamic_settings import dynamic_settings
from src.core.settings import settings
from src.core.volatility_tracker import (
    detect_steam_move,
    get_volatility,
    record_odds_snapshot,
)
from src.data.redis_cache import cache
from src.integrations.odds_fetcher import (
    OddsFetcher,
    get_markets_for_sport,
    get_trading_window_end,
)

log = logging.getLogger(__name__)

SNAPSHOT_KEY = "scout:odds_snapshot"
ALERTS_KEY = "scout:alerts"
KICKOFF_TIMES_KEY = "orchestrator:kickoff_times"


class ScoutAgent:
    """Monitors OddsAPI for steam moves and aggregates injury intelligence."""

    def __init__(self) -> None:
        self.odds_fetcher = OddsFetcher()
        self._previous_snapshot: Dict[str, Dict[str, float]] = {}

    async def monitor_odds(self) -> List[Dict[str, Any]]:
        """Poll odds and detect steam moves. Returns list of alert dicts."""
        alerts: List[Dict[str, Any]] = []

        # Load previous snapshot from Redis
        prev = cache.get_json(SNAPSHOT_KEY) or {}
        self._previous_snapshot = prev

        # Use dynamic settings for active sports (fallback to env var)
        base_sports = dynamic_settings.get_active_sports()
        if not base_sports:
            base_sports = [s.strip() for s in settings.live_sports.split(",") if s.strip()]

        # Resolve base keys to exact in-season API keys via /v4/sports
        try:
            api_active = await self.odds_fetcher.get_active_sports_from_api_async()
        except Exception:
            api_active = []
        active_sports = OddsFetcher.resolve_sport_keys(base_sports, api_active)

        current_snapshot: Dict[str, Dict[str, float]] = {}
        kickoff_times: List[str] = []

        # Trading Session Window: scope API to "now → tomorrow 06:59 UTC"
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        window_end = get_trading_window_end()
        window_end_iso = window_end.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Fetch 12h-ago momentum data per sport
        momentum_data: Dict[str, Dict[str, float]] = {}
        for sport in active_sports:
            # Deep Sleep: skip sports with no events in today's window
            if self.odds_fetcher.is_sport_deep_sleeping(sport):
                log.debug("Skipping %s (deep-sleep, no events in trading window)", sport)
                continue
            try:
                sport_momentum = self.odds_fetcher.get_odds_12h_ago(sport)
                if sport_momentum:
                    for eid, sels in sport_momentum.items():
                        for sel, ip_val in sels.items():
                            momentum_data[f"{eid}:{sel}"] = ip_val
            except Exception:
                pass

        now_utc = datetime.now(timezone.utc)

        for sport in active_sports:
            # Deep Sleep check
            if self.odds_fetcher.is_sport_deep_sleeping(sport):
                continue

            # Dynamic Market Pruning: only request markets with sharp-book value
            sport_markets = get_markets_for_sport(sport)

            try:
                events = await self.odds_fetcher.get_sport_odds_async(
                    sport_key=sport, regions="eu", markets=sport_markets,
                    ttl_seconds=60,
                    commence_time_from=now_iso,
                    commence_time_to=window_end_iso,
                )
            except Exception:
                continue
            if not isinstance(events, list):
                continue

            # Deep Sleep: if no events in the trading window, hibernate this sport
            if len(events) == 0:
                self.odds_fetcher.mark_sport_deep_sleep(sport)
                continue

            for event in events:
                event_id = str(event.get("id") or "")
                home = event.get("home_team") or ""
                away = event.get("away_team") or ""
                # Collect kickoff times for adaptive polling
                commence = event.get("commence_time")
                if commence:
                    kickoff_times.append(str(commence))
                    # Live-event filter: skip events that have already kicked off.
                    # In-play odds have fundamentally different dynamics — our
                    # pre-match model has zero edge on live markets.
                    try:
                        ct = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                        if ct <= now_utc:
                            continue
                    except (ValueError, TypeError):
                        pass
                if not event_id or not home:
                    continue

                # Extract current odds from sharp books (h2h + totals)
                for bm in event.get("bookmakers", []):
                    if bm.get("key") not in ("pinnacle", "betfair_ex_uk"):
                        continue
                    for mkt in bm.get("markets", []):
                        mkt_key = mkt.get("key") or ""
                        if mkt_key not in ("h2h", "totals"):
                            continue
                        for outcome in mkt.get("outcomes", []):
                            name = outcome.get("name")
                            price = outcome.get("price")
                            point = outcome.get("point")
                            if not name or not price:
                                continue

                            # Build unique key (includes point for totals)
                            if mkt_key == "totals" and point is not None:
                                label = f"{name}|{point}"
                                snap_key = f"{event_id}:{label}"
                            else:
                                label = name
                                snap_key = f"{event_id}:{name}"

                            current_odds = float(price)
                            current_ip = 1.0 / current_odds if current_odds > 1.0 else 0.0
                            current_snapshot[snap_key] = {"odds": current_odds, "ip": current_ip}

                            # Record for volatility tracking
                            try:
                                record_odds_snapshot(label, current_ip)
                            except Exception:
                                pass

                            # Check for steam move
                            prev_data = prev.get(snap_key, {})
                            prev_odds = prev_data.get("odds", current_odds)

                            team_vol = get_volatility(label)
                            is_steam = detect_steam_move(current_odds, prev_odds, team_vol)

                            if is_steam:
                                # Compute 12h momentum if available
                                hist_ip = momentum_data.get(snap_key)
                                market_momentum = round(current_ip - hist_ip, 4) if hist_ip else 0.0

                                alert_type = "totals_steam" if mkt_key == "totals" else "steam_move"
                                selection_label = label if mkt_key == "totals" else name

                                alerts.append({
                                    "type": alert_type,
                                    "event_id": event_id,
                                    "sport": sport,
                                    "home": home,
                                    "away": away,
                                    "selection": selection_label,
                                    "market": mkt_key,
                                    "commence_time": str(commence or ""),
                                    "prev_odds": prev_odds,
                                    "current_odds": current_odds,
                                    "movement_pct": round(abs(1.0 / current_odds - 1.0 / prev_odds) * 100, 2),
                                    "team_volatility": team_vol,
                                    "market_momentum": market_momentum,
                                })
                                log.info(
                                    "%s detected: %s %s %.2f -> %.2f",
                                    alert_type, sport, selection_label, prev_odds, current_odds,
                                )

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

    async def monitor_injuries(self) -> List[Dict[str, Any]]:
        """Monitor for breaking injury news via aggregated sources.

        Uses the unified injury aggregator (API-Sports + Rotowire RSS + LLM)
        to detect key player absences for today's events.
        """
        alerts: List[Dict[str, Any]] = []

        # Get teams from currently tracked events (use resolved keys)
        base_sports = dynamic_settings.get_active_sports()
        if not base_sports:
            base_sports = [s.strip() for s in settings.live_sports.split(",") if s.strip()]
        try:
            api_active = await self.odds_fetcher.get_active_sports_from_api_async()
        except Exception:
            api_active = []
        sports = OddsFetcher.resolve_sport_keys(base_sports, api_active)

        # Collect today's events
        events_to_check: List[Dict[str, str]] = []
        for sport in sports:
            try:
                events = await self.odds_fetcher.get_sport_odds_async(
                    sport_key=sport, regions="eu", markets="h2h", ttl_seconds=300,
                )
                if isinstance(events, list):
                    for e in events:
                        home = e.get("home_team", "")
                        away = e.get("away_team", "")
                        if home and away:
                            events_to_check.append({
                                "home": home,
                                "away": away,
                                "sport": sport,
                                "event_id": str(e.get("id", "")),
                                "commence_time": str(e.get("commence_time", "")),
                            })
            except Exception:
                continue

        # Run injury aggregation for each event (cap at 15 events)
        from src.integrations.injury_aggregator import aggregate_injury_intel

        for ev in events_to_check[:15]:
            try:
                result = await aggregate_injury_intel(
                    home_team=ev["home"],
                    away_team=ev["away"],
                    sport=ev["sport"],
                    event_date=ev.get("commence_time", ""),
                )
                injuries = result.get("injuries", [])
                if injuries:
                    # Check for confirmed "Out" status players
                    out_players = [
                        inj for inj in injuries
                        if inj.get("status", "").lower() == "out"
                    ]
                    if out_players:
                        alerts.append({
                            "type": "breaking_injury",
                            "event_id": ev["event_id"],
                            "sport": ev["sport"],
                            "home": ev["home"],
                            "away": ev["away"],
                            "injuries": injuries,
                            "out_players": out_players,
                            "sources": result.get("sources", []),
                            "text": "; ".join(
                                f"{p['player']} ({p['team']}): {p['status']}"
                                for p in out_players
                            ),
                        })
                        log.info(
                            "Breaking injury detected: %s vs %s — %d players out",
                            ev["home"], ev["away"], len(out_players),
                        )
            except Exception as exc:
                log.warning("Injury aggregation failed for %s vs %s: %s", ev["home"], ev["away"], exc)

        return alerts

    @staticmethod
    def get_recent_alerts() -> List[Dict[str, Any]]:
        """Retrieve recent scout alerts from Redis."""
        return cache.get_json(ALERTS_KEY) or []
