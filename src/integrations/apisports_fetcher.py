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

# API-Sports sport-specific base URLs
API_BASKETBALL_BASE = "https://v1.basketball.api-sports.io"
API_HOCKEY_BASE = "https://v1.hockey.api-sports.io"


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
    # API-Football predictions (1X2 probabilities)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_percent_to_prob(raw_val: Any) -> float:
        """Parse values like "45%" -> 0.45 with safe fallback."""
        try:
            s = str(raw_val or "0").replace("%", "").strip()
            return max(0.0, min(1.0, float(s) / 100.0))
        except Exception:
            return 0.0

    async def _get_predictions_raw(self, base_url: str, query_key: str, event_id: int) -> Dict[str, Any]:
        """Fetch raw predictions from a specific API-Sports host."""
        import httpx
        from src.integrations.base_fetcher import build_httpx_ssl_verify

        headers = {
            "x-apisports-key": settings.apisports_api_key,
            "Accept": "application/json",
        }
        url = f"{base_url}/predictions"
        params = {query_key: int(event_id)}
        async with httpx.AsyncClient(timeout=20, verify=build_httpx_ssl_verify()) as client:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code} {url}")
            return resp.json()

    async def get_predictions_by_event_id_async(
        self,
        event_id: int,
        sport: str,
        ttl_seconds: int = 43200,
    ) -> Dict[str, float]:
        """Fetch API-Sports prediction percentages by event/game/fixture id.

        Sport routing:
        - soccer*     -> football host, query `fixture=<id>`
        - basketball* -> basketball host, query `game=<id>`
        - icehockey*  -> hockey host, query `game=<id>`
        """
        sport_key = str(sport or "").lower()
        cache_key = f"apisports:predictions:{sport_key}:{int(event_id)}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        if sport_key.startswith("soccer"):
            base_url, query_key = settings.apisports_base_url, "fixture"
        elif sport_key.startswith("basketball"):
            base_url, query_key = API_BASKETBALL_BASE, "game"
        elif sport_key.startswith("icehockey"):
            base_url, query_key = API_HOCKEY_BASE, "game"
        else:
            cache.set_json(cache_key, {}, ttl_seconds)
            return {}

        try:
            data = await self._get_predictions_raw(base_url, query_key, int(event_id))
            response_list = data.get("response") or []
            if not response_list:
                cache.set_json(cache_key, {}, ttl_seconds)
                return {}

            percent_data = (
                response_list[0]
                .get("predictions", {})
                .get("percent", {})
            )

            # US sports often have no draw field; keep neutral fallback = 0.0
            result = {
                "api_prob_home": self._parse_percent_to_prob(percent_data.get("home", "0%")),
                "api_prob_draw": self._parse_percent_to_prob(percent_data.get("draw", "0%")),
                "api_prob_away": self._parse_percent_to_prob(percent_data.get("away", "0%")),
            }
            cache.set_json(cache_key, result, ttl_seconds)
            log.info("API-Sports predictions loaded sport=%s id=%s %s", sport_key, event_id, result)
            return result
        except Exception as exc:
            log.warning("API-Sports predictions fetch failed sport=%s id=%s: %s", sport_key, event_id, exc)
            cache.set_json(cache_key, {}, 300)
            return {}

    async def get_predictions_by_fixture_id_async(
        self, fixture_id: int, ttl_seconds: int = 43200
    ) -> Dict[str, float]:
        """Backward-compatible soccer wrapper (fixture id on football host)."""
        return await self.get_predictions_by_event_id_async(fixture_id, sport="soccer", ttl_seconds=ttl_seconds)

    def get_predictions_by_fixture_id(self, fixture_id: int, ttl_seconds: int = 43200) -> Dict[str, float]:
        """Sync wrapper for get_predictions_by_fixture_id_async."""
        return _safe_sync_run(self.get_predictions_by_fixture_id_async(fixture_id, ttl_seconds))

    async def get_predictions_for_event_async(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        event_date: str = "",
        ttl_seconds: int = 43200,
    ) -> Dict[str, float]:
        """Fetch API predictions for a soccer/basketball/hockey event.

        Routes to sport-specific API-Sports host and endpoint.
        """
        sport_key = str(sport or "").lower()
        if not (
            sport_key.startswith("soccer")
            or sport_key.startswith("basketball")
            or sport_key.startswith("icehockey")
        ):
            return {}

        cache_key = f"apisports:pred:event:{sport_key}:{home_team}:{away_team}:{self._resolve_date(event_date)}".lower().replace(" ", "_")
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached

        try:
            date_str = self._resolve_date(event_date)

            # Soccer uses existing wrapper + /fixtures endpoint
            if sport_key.startswith("soccer"):
                fixtures_data = await self.get("fixtures", params={"date": date_str})
                event_ids = self._match_fixture_ids(fixtures_data.get("response") or [], home_team, away_team)
            else:
                # Basketball / Hockey use dedicated hosts + /games endpoint
                import httpx
                from src.integrations.base_fetcher import build_httpx_ssl_verify

                if sport_key.startswith("basketball"):
                    base_url = API_BASKETBALL_BASE
                else:
                    base_url = API_HOCKEY_BASE

                headers = {
                    "x-apisports-key": settings.apisports_api_key,
                    "Accept": "application/json",
                }
                url = f"{base_url}/games"
                params = {"date": date_str}
                async with httpx.AsyncClient(timeout=20, verify=build_httpx_ssl_verify()) as client:
                    resp = await client.get(url, headers=headers, params=params)
                    if resp.status_code != 200:
                        cache.set_json(cache_key, {}, 300)
                        return {}
                    games_data = resp.json()
                event_ids = self._match_game_ids(games_data.get("response") or [], home_team, away_team)

            if not event_ids:
                cache.set_json(cache_key, {}, ttl_seconds)
                return {}

            preds = await self.get_predictions_by_event_id_async(int(event_ids[0]), sport=sport_key, ttl_seconds=ttl_seconds)
            cache.set_json(cache_key, preds, ttl_seconds)
            return preds
        except Exception as exc:
            log.warning("API-Sports event prediction fetch failed for %s vs %s (%s): %s", home_team, away_team, sport_key, exc)
            cache.set_json(cache_key, {}, 300)
            return {}

    def get_predictions_for_event(
        self,
        home_team: str,
        away_team: str,
        sport: str,
        event_date: str = "",
        ttl_seconds: int = 43200,
    ) -> Dict[str, float]:
        return _safe_sync_run(
            self.get_predictions_for_event_async(home_team, away_team, sport, event_date, ttl_seconds)
        )

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

            from src.integrations.base_fetcher import build_httpx_ssl_verify
            async with httpx.AsyncClient(timeout=20, verify=build_httpx_ssl_verify()) as client:
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
    def _match_game_ids(games: list, home_team: str, away_team: str) -> List[int]:
        """Find basketball/hockey game IDs via normalized team names."""
        ids: List[int] = []
        norm_home = normalize_team(home_team)
        norm_away = normalize_team(away_team)

        for g in games:
            teams = g.get("teams", {}) or {}
            # API-Basketball sometimes uses visitors/home, hockey often home/away
            g_home = normalize_team(
                (teams.get("home") or {}).get("name")
                or (teams.get("local") or {}).get("name")
                or ""
            )
            g_away = normalize_team(
                (teams.get("away") or {}).get("name")
                or (teams.get("visitors") or {}).get("name")
                or ""
            )
            if g_home == norm_home and g_away == norm_away:
                gid = g.get("id") or (g.get("game") or {}).get("id") or (g.get("fixture") or {}).get("id")
                if gid:
                    ids.append(int(gid))
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
