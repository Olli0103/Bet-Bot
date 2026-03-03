"""API-Sports fetcher for official injuries and fixture data.

Supports soccer injuries natively via /injuries endpoint.
For other sports, returns empty lists (fallback to RSS/Reddit).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.core.settings import settings
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)


class APISportsFetcher(AsyncBaseFetcher):
    def __init__(self):
        headers = {
            "x-apisports-key": settings.apisports_api_key,
            "Accept": "application/json",
        }
        super().__init__(base_url=settings.apisports_base_url, headers=headers)

    # ------------------------------------------------------------------
    # Existing fixture/injury methods
    # ------------------------------------------------------------------

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
        return _safe_sync_run(self.get_fixtures_by_ids_async(fixture_ids, ttl_seconds))

    def get_injuries_by_fixture_ids(self, fixture_ids: List[int], ttl_seconds: int = 180) -> Dict[str, Any]:
        return _safe_sync_run(self.get_injuries_by_fixture_ids_async(fixture_ids, ttl_seconds))

    # ------------------------------------------------------------------
    # EPIC 1: Official injury data for today's events
    # ------------------------------------------------------------------

    async def get_injuries_for_event_async(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        event_date: str = "",
        ttl_seconds: int = 300,
    ) -> List[Dict[str, str]]:
        """Fetch official injuries for a specific match.

        For soccer: uses /injuries endpoint via fixture ID lookup.
        For other sports: API-Sports doesn't have a universal injury endpoint,
        so we return an empty list and rely on RSS/Reddit fallback.

        Returns a list of dicts: [{"player": "...", "status": "Out|Doubtful|..."}]
        """
        if not settings.apisports_api_key:
            return []

        # Only soccer has reliable injury endpoint on API-Sports
        if not sport.startswith("soccer"):
            return []

        cache_key = f"apisports:injuries:match:{home_team}:{away_team}".lower().replace(" ", "_")
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        try:
            date_str = event_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

            # Step 1: Find fixture IDs for today matching these teams
            fixtures_data = await self.get("fixtures", params={"date": date_str})
            fixtures = fixtures_data.get("response") or []

            fixture_ids: List[int] = []
            for fx in fixtures:
                teams = fx.get("teams", {})
                fx_home = (teams.get("home", {}).get("name") or "").lower()
                fx_away = (teams.get("away", {}).get("name") or "").lower()
                if (home_team.lower() in fx_home or fx_home in home_team.lower() or
                        away_team.lower() in fx_away or fx_away in away_team.lower()):
                    fid = fx.get("fixture", {}).get("id")
                    if fid:
                        fixture_ids.append(int(fid))

            if not fixture_ids:
                cache.set_json(cache_key, [], ttl_seconds)
                return []

            # Step 2: Fetch injuries for matched fixtures
            injuries_data = await self.get_injuries_by_fixture_ids_async(
                fixture_ids[:5], ttl_seconds=ttl_seconds
            )
            raw_injuries = injuries_data.get("response") or []

            result: List[Dict[str, str]] = []
            for inj in raw_injuries:
                player = inj.get("player", {})
                player_name = player.get("name") or "Unknown"
                reason = player.get("reason") or inj.get("player", {}).get("type") or "Unknown"
                team_name = inj.get("team", {}).get("name") or ""
                result.append({
                    "player": player_name,
                    "team": team_name,
                    "status": reason,
                    "source": "api-sports",
                })

            cache.set_json(cache_key, result, ttl_seconds)
            return result

        except Exception as exc:
            log.warning("API-Sports injury fetch failed for %s vs %s: %s", home_team, away_team, exc)
            cache.set_json(cache_key, [], ttl_seconds)
            return []

    def get_injuries_for_event(
        self, home_team: str, away_team: str, sport: str, event_date: str = ""
    ) -> List[Dict[str, str]]:
        """Sync wrapper for get_injuries_for_event_async."""
        return _safe_sync_run(
            self.get_injuries_for_event_async(home_team, away_team, sport, event_date)
        )
