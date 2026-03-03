"""football-data.org integration — soccer matches, standings, team stats.

Free tier (v4):
- 12 competitions: PL, BL1, SA, FL1, PD, ELC, DED, PPL, BSA, CL, EC, WC
- Rate limit: 10 requests/minute
- Auth: X-Auth-Token header

All responses are cached in Redis to respect the strict rate limit.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# Map OddsAPI sport keys → football-data.org competition codes
COMPETITION_CODES: Dict[str, str] = {
    "soccer_epl": "PL",
    "soccer_germany_bundesliga": "BL1",
    "soccer_spain_la_liga": "PD",
    "soccer_italy_serie_a": "SA",
    "soccer_france_ligue_one": "FL1",
    "soccer_england_league1": "ELC",
    "soccer_netherlands_eredivisie": "DED",
    "soccer_portugal_primeira_liga": "PPL",
    "soccer_brazil_serie_a": "BSA",
    "soccer_uefa_champs_league": "CL",
}


class FootballDataFetcher:
    """Client for football-data.org v4 API."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        headers = {}
        if api_key:
            headers["X-Auth-Token"] = api_key
        self._fetcher = AsyncBaseFetcher(
            base_url=BASE_URL,
            headers=headers,
            timeout=15,
        )

    # ------------------------------------------------------------------
    # Low-level cached GET
    # ------------------------------------------------------------------

    def _get_cached(self, path: str, params: Optional[Dict[str, Any]] = None, ttl: int = 3600) -> Any:
        """GET with Redis caching."""
        norm_params = re.sub(r"[^a-z0-9]", "", str(params or {}).lower())
        cache_key = f"footballdata:{path.replace('/', ':')}:{norm_params}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached
        try:
            result = _safe_sync_run(self._fetcher.get(path, params or {}), timeout=20)
            if result is not None:
                cache.set_json(cache_key, result, ttl_seconds=ttl)
            return result
        except Exception as exc:
            log.warning("football-data.org fetch failed %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Matches / results
    # ------------------------------------------------------------------

    def get_matches(
        self, sport_key: str, status: str = "FINISHED", matchday: Optional[int] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get matches for a competition.

        Parameters
        ----------
        sport_key : str
            OddsAPI-style sport key (e.g. "soccer_epl").
        status : str
            Filter: FINISHED, SCHEDULED, IN_PLAY, etc.
        matchday : int, optional
            Filter to a specific matchday.
        limit : int
            Max results.
        """
        code = COMPETITION_CODES.get(sport_key)
        if not code:
            return []
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        if matchday:
            params["matchday"] = matchday

        data = self._get_cached(
            f"/competitions/{code}/matches",
            params,
            ttl=1800,  # 30 min
        )
        if not data:
            return []
        matches = data.get("matches") or []
        return [self._normalize_match(m) for m in matches[:limit]]

    def get_team_matches(self, team_id: int, status: str = "FINISHED", limit: int = 15) -> List[Dict[str, Any]]:
        """Get matches for a specific team (by football-data team ID)."""
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        params["limit"] = limit

        data = self._get_cached(
            f"/teams/{team_id}/matches",
            params,
            ttl=1800,
        )
        if not data:
            return []
        matches = data.get("matches") or []
        return [self._normalize_match(m) for m in matches[:limit]]

    # ------------------------------------------------------------------
    # Standings
    # ------------------------------------------------------------------

    def get_standings(self, sport_key: str, table_type: str = "TOTAL") -> List[Dict[str, Any]]:
        """Get league standings.

        Parameters
        ----------
        sport_key : str
            OddsAPI-style sport key.
        table_type : str
            "TOTAL", "HOME", or "AWAY".
        """
        code = COMPETITION_CODES.get(sport_key)
        if not code:
            return []

        data = self._get_cached(
            f"/competitions/{code}/standings",
            {},
            ttl=3600,
        )
        if not data:
            return []

        for standing in data.get("standings") or []:
            if standing.get("type") == table_type:
                table = standing.get("table") or []
                return [self._normalize_standing(row) for row in table]
        return []

    def get_home_away_standings(self, sport_key: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get HOME and AWAY standings — useful for home/away splits."""
        return {
            "home": self.get_standings(sport_key, "HOME"),
            "away": self.get_standings(sport_key, "AWAY"),
        }

    # ------------------------------------------------------------------
    # Top scorers
    # ------------------------------------------------------------------

    def get_scorers(self, sport_key: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top scorers for a competition."""
        code = COMPETITION_CODES.get(sport_key)
        if not code:
            return []
        data = self._get_cached(
            f"/competitions/{code}/scorers",
            {"limit": limit},
            ttl=3600,
        )
        if not data:
            return []
        scorers = data.get("scorers") or []
        return [
            {
                "player": s.get("player", {}).get("name", ""),
                "team": s.get("team", {}).get("name", ""),
                "goals": s.get("goals", 0),
                "assists": s.get("assists", 0),
                "penalties": s.get("penalties", 0),
            }
            for s in scorers
        ]

    # ------------------------------------------------------------------
    # Normalizers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_match(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a football-data.org match to a standard dict."""
        score = raw.get("score") or {}
        ft = score.get("fullTime") or {}
        ht = score.get("halfTime") or {}
        home_team = raw.get("homeTeam") or {}
        away_team = raw.get("awayTeam") or {}

        return {
            "match_id": raw.get("id"),
            "date": str(raw.get("utcDate", "")),
            "matchday": raw.get("matchday"),
            "status": raw.get("status", ""),
            "home_team": home_team.get("name", ""),
            "home_team_short": home_team.get("shortName", ""),
            "home_team_id": home_team.get("id"),
            "away_team": away_team.get("name", ""),
            "away_team_short": away_team.get("shortName", ""),
            "away_team_id": away_team.get("id"),
            "home_score": ft.get("home"),
            "away_score": ft.get("away"),
            "home_ht_score": ht.get("home"),
            "away_ht_score": ht.get("away"),
            "winner": score.get("winner", ""),
            "source": "football-data.org",
        }

    @staticmethod
    def _normalize_standing(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a football-data.org table row."""
        team = raw.get("team") or {}
        return {
            "team": team.get("name", ""),
            "team_short": team.get("shortName", ""),
            "team_id": team.get("id"),
            "position": raw.get("position", 0),
            "played": raw.get("playedGames", 0),
            "won": raw.get("won", 0),
            "drawn": raw.get("draw", 0),
            "lost": raw.get("lost", 0),
            "goals_for": raw.get("goalsFor", 0),
            "goals_against": raw.get("goalsAgainst", 0),
            "goal_diff": raw.get("goalDifference", 0),
            "points": raw.get("points", 0),
            "form": raw.get("form", ""),
            "source": "football-data.org",
        }
