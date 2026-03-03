"""API-Sports fetcher for official injuries and fixture data.

Supports soccer injuries via /injuries endpoint and basketball injuries
via the API-Basketball /injuries endpoint. For other sports, returns
empty lists (fallback to Rotowire RSS).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from src.core.settings import settings
from src.core.sport_mapping import normalize_team
from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)

# API-Basketball base URL (separate host from football API)
API_BASKETBALL_BASE = "https://v1.basketball.api-sports.io"


class APISportsFetcher(AsyncBaseFetcher):
    def __init__(self):
        headers = {
            "x-apisports-key": settings.apisports_api_key,
            "Accept": "application/json",
        }
        super().__init__(base_url=settings.apisports_base_url, headers=headers)

    # ------------------------------------------------------------------
    # Existing fixture/injury methods (soccer)
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
    # Official injury data for today's events
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

        - Soccer: /injuries endpoint via fixture ID lookup.
        - Basketball: /injuries endpoint via API-Basketball host.
        - Other sports: returns empty list (fallback to Rotowire RSS).

        Returns a list of dicts: [{"player": "...", "team": "...", "status": "...", "source": "api-sports"}]
        """
        if not settings.apisports_api_key:
            return []

        if sport.startswith("soccer"):
            return await self._soccer_injuries(home_team, away_team, event_date, ttl_seconds)
        if sport.startswith("basketball"):
            return await self._basketball_injuries(home_team, away_team, event_date, ttl_seconds)

        # NFL, NHL, Tennis — no API-Sports endpoint; rely on RSS
        return []

    async def _soccer_injuries(
        self, home_team: str, away_team: str, event_date: str, ttl_seconds: int
    ) -> List[Dict[str, str]]:
        """Soccer injuries via v3.football.api-sports.io /injuries."""
        cache_key = f"apisports:inj:soccer:{home_team}:{away_team}".lower().replace(" ", "_")
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        try:
            date_str = self._resolve_date(event_date)

            # Step 1: find fixture IDs matching these teams today
            fixtures_data = await self.get("fixtures", params={"date": date_str})
            fixture_ids = self._match_fixture_ids(
                fixtures_data.get("response") or [], home_team, away_team
            )
            if not fixture_ids:
                cache.set_json(cache_key, [], ttl_seconds)
                return []

            # Step 2: fetch injuries for those fixtures
            injuries_data = await self.get_injuries_by_fixture_ids_async(
                fixture_ids[:5], ttl_seconds=ttl_seconds
            )
            result = self._parse_football_injuries(injuries_data.get("response") or [])
            cache.set_json(cache_key, result, ttl_seconds)
            return result

        except Exception as exc:
            log.warning("API-Sports soccer injury fetch failed for %s vs %s: %s", home_team, away_team, exc)
            cache.set_json(cache_key, [], ttl_seconds)
            return []

    async def _basketball_injuries(
        self, home_team: str, away_team: str, event_date: str, ttl_seconds: int
    ) -> List[Dict[str, str]]:
        """Basketball injuries via v1.basketball.api-sports.io /injuries.

        The API-Basketball /injuries endpoint accepts a date filter.
        We query by date to get all injuries for today, then filter by team.
        """
        cache_key = f"apisports:inj:bball:{home_team}:{away_team}".lower().replace(" ", "_")
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        try:
            import httpx
            date_str = self._resolve_date(event_date)
            headers = {
                "x-apisports-key": settings.apisports_api_key,
                "Accept": "application/json",
            }
            url = f"{API_BASKETBALL_BASE}/injuries"
            params = {"date": date_str}

            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code != 200:
                    cache.set_json(cache_key, [], ttl_seconds)
                    return []
                data = resp.json()

            raw = data.get("response") or []
            result: List[Dict[str, str]] = []
            norm_home = normalize_team(home_team)
            norm_away = normalize_team(away_team)

            for entry in raw:
                team_name = entry.get("team", {}).get("name") or ""
                norm_entry = normalize_team(team_name)
                if norm_entry != norm_home and norm_entry != norm_away:
                    continue
                player_name = entry.get("player", {}).get("name") or "Unknown"
                status = entry.get("status") or entry.get("player", {}).get("type") or "Unknown"
                result.append({
                    "player": player_name,
                    "team": entry.get("team", {}).get("name") or "",
                    "status": status,
                    "source": "api-sports",
                })

            cache.set_json(cache_key, result, ttl_seconds)
            return result

        except Exception as exc:
            log.warning("API-Sports basketball injury fetch failed for %s vs %s: %s", home_team, away_team, exc)
            cache.set_json(cache_key, [], ttl_seconds)
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_date(event_date: str) -> str:
        """Normalise an event date string to YYYY-MM-DD."""
        if not event_date:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            dt = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _match_fixture_ids(
        fixtures: list, home_team: str, away_team: str
    ) -> List[int]:
        """Find fixture IDs where the team names match via canonical normalization."""
        ids: List[int] = []
        norm_home = normalize_team(home_team)
        norm_away = normalize_team(away_team)
        for fx in fixtures:
            teams = fx.get("teams", {})
            fx_home = normalize_team(teams.get("home", {}).get("name") or "")
            fx_away = normalize_team(teams.get("away", {}).get("name") or "")
            if fx_home == norm_home and fx_away == norm_away:
                fid = fx.get("fixture", {}).get("id")
                if fid:
                    ids.append(int(fid))
        return ids

    @staticmethod
    def _parse_football_injuries(raw: list) -> List[Dict[str, str]]:
        """Parse the API-Football /injuries response format."""
        result: List[Dict[str, str]] = []
        for inj in raw:
            player = inj.get("player", {})
            result.append({
                "player": player.get("name") or "Unknown",
                "team": inj.get("team", {}).get("name") or "",
                "status": player.get("reason") or player.get("type") or "Unknown",
                "source": "api-sports",
            })
        return result

    def get_injuries_for_event(
        self, home_team: str, away_team: str, sport: str, event_date: str = ""
    ) -> List[Dict[str, str]]:
        """Sync wrapper for get_injuries_for_event_async."""
        return _safe_sync_run(
            self.get_injuries_for_event_async(home_team, away_team, sport, event_date)
        )
