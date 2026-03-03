"""TheSportsDB integration — free tier for match results, standings, team info.

Endpoints used (v1, free API key = 1):
- Past events by league:  /eventspastrounds.php?id={league_id}&r={rounds}
- Last 5 events by team:  /eventslast.php?id={team_id}
- Next 5 events by team:  /eventsnext.php?id={team_id}
- League table:           /lookuptable.php?l={league_id}&s={season}
- Team details:           /searchteams.php?t={team_name}
- Team by ID:             /lookupteam.php?id={team_id}

All responses are cached in Redis to respect rate limits (free tier: ~30 req/min).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.data.redis_cache import cache
from src.integrations.base_fetcher import AsyncBaseFetcher, _safe_sync_run

log = logging.getLogger(__name__)

# Free-tier key (public, rate-limited)
DEFAULT_API_KEY = "3"
BASE_URL = "https://www.thesportsdb.com/api/v1/json"

# League ID mappings (TheSportsDB → OddsAPI sport key)
LEAGUE_IDS: Dict[str, int] = {
    "soccer_epl": 4328,
    "soccer_germany_bundesliga": 4331,
    "soccer_spain_la_liga": 4335,
    "soccer_italy_serie_a": 4332,
    "soccer_france_ligue_one": 4334,
    "soccer_germany_bundesliga2": 4346,
    "basketball_nba": 4387,
    "icehockey_nhl": 4380,
    "americanfootball_nfl": 4391,
}

# Season format varies: soccer uses "2025-2026", NBA uses "2025"
LEAGUE_SEASON_FORMAT: Dict[str, str] = {
    "soccer_epl": "2025-2026",
    "soccer_germany_bundesliga": "2025-2026",
    "soccer_spain_la_liga": "2025-2026",
    "soccer_italy_serie_a": "2025-2026",
    "soccer_france_ligue_one": "2025-2026",
    "soccer_germany_bundesliga2": "2025-2026",
    "basketball_nba": "2025-2026",
    "icehockey_nhl": "2025-2026",
    "americanfootball_nfl": "2025",
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


class SportsDBFetcher:
    """Client for TheSportsDB free-tier API."""

    def __init__(self, api_key: str = DEFAULT_API_KEY) -> None:
        self.api_key = api_key
        self._fetcher = AsyncBaseFetcher(
            base_url=f"{BASE_URL}/{api_key}",
            timeout=15,
        )

    # ------------------------------------------------------------------
    # Low-level cached GET
    # ------------------------------------------------------------------

    def _get_cached(self, path: str, params: Dict[str, Any], ttl: int = 3600) -> Any:
        """GET with Redis caching."""
        cache_key = f"sportsdb:{path}:{_normalize(str(params))}"
        cached = cache.get_json(cache_key)
        if cached is not None:
            return cached
        try:
            result = _safe_sync_run(self._fetcher.get(path, params), timeout=20)
            if result is not None:
                cache.set_json(cache_key, result, ttl_seconds=ttl)
            return result
        except Exception as exc:
            log.warning("SportsDB fetch failed %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Team search
    # ------------------------------------------------------------------

    def search_team(self, team_name: str) -> Optional[Dict[str, Any]]:
        """Search for a team by name. Returns first match or None."""
        data = self._get_cached(
            "/searchteams.php", {"t": team_name}, ttl=86400
        )
        if not data:
            return None
        teams = data.get("teams") or []
        return teams[0] if teams else None

    def get_team_id(self, team_name: str) -> Optional[int]:
        """Resolve team name to TheSportsDB team ID."""
        team = self.search_team(team_name)
        if team:
            return int(team.get("idTeam", 0)) or None
        return None

    # ------------------------------------------------------------------
    # Past events (match results)
    # ------------------------------------------------------------------

    def get_past_events(
        self, sport_key: str, rounds: int = 10
    ) -> List[Dict[str, Any]]:
        """Get past events for a league (last N rounds)."""
        league_id = LEAGUE_IDS.get(sport_key)
        if not league_id:
            return []
        data = self._get_cached(
            "/eventspastrounds.php",
            {"id": league_id, "r": rounds},
            ttl=1800,  # 30 min cache
        )
        if not data:
            return []
        events = data.get("events") or []
        return [self._normalize_event(e) for e in events]

    def get_team_last_events(self, team_id: int) -> List[Dict[str, Any]]:
        """Get last 5 events for a team by ID."""
        data = self._get_cached(
            "/eventslast.php", {"id": team_id}, ttl=1800
        )
        if not data:
            return []
        events = data.get("results") or []
        return [self._normalize_event(e) for e in events]

    def get_team_next_events(self, team_id: int) -> List[Dict[str, Any]]:
        """Get next 5 upcoming events for a team by ID."""
        data = self._get_cached(
            "/eventsnext.php", {"id": team_id}, ttl=1800
        )
        if not data:
            return []
        events = data.get("events") or []
        return [self._normalize_event(e) for e in events]

    # ------------------------------------------------------------------
    # League standings / table
    # ------------------------------------------------------------------

    def get_league_table(self, sport_key: str) -> List[Dict[str, Any]]:
        """Get current league standings."""
        league_id = LEAGUE_IDS.get(sport_key)
        season = LEAGUE_SEASON_FORMAT.get(sport_key, "2025-2026")
        if not league_id:
            return []
        data = self._get_cached(
            "/lookuptable.php",
            {"l": league_id, "s": season},
            ttl=3600,  # 1 hour cache
        )
        if not data:
            return []
        table = data.get("table") or []
        return [self._normalize_standing(row) for row in table]

    # ------------------------------------------------------------------
    # Normalizers — produce consistent DTOs regardless of API format
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_event(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a TheSportsDB event to a standard dict."""
        home_score = raw.get("intHomeScore")
        away_score = raw.get("intAwayScore")
        return {
            "event_id": str(raw.get("idEvent", "")),
            "date": str(raw.get("dateEvent", "")),
            "time": str(raw.get("strTime", "")),
            "home_team": str(raw.get("strHomeTeam", "")),
            "away_team": str(raw.get("strAwayTeam", "")),
            "home_score": int(home_score) if home_score is not None else None,
            "away_score": int(away_score) if away_score is not None else None,
            "league": str(raw.get("strLeague", "")),
            "season": str(raw.get("strSeason", "")),
            "venue": str(raw.get("strVenue", "")),
            "status": str(raw.get("strStatus", "")),
            "home_shots": _safe_int(raw.get("intHomeShots")),
            "away_shots": _safe_int(raw.get("intAwayShots")),
            "round": str(raw.get("intRound", "")),
            "source": "thesportsdb",
        }

    @staticmethod
    def _normalize_standing(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a TheSportsDB table row to a standard dict."""
        return {
            "team": str(raw.get("strTeam", "")),
            "team_id": str(raw.get("idTeam", "")),
            "position": _safe_int(raw.get("intRank")),
            "played": _safe_int(raw.get("intPlayed")),
            "won": _safe_int(raw.get("intWin")),
            "drawn": _safe_int(raw.get("intDraw")),
            "lost": _safe_int(raw.get("intLoss")),
            "goals_for": _safe_int(raw.get("intGoalsFor")),
            "goals_against": _safe_int(raw.get("intGoalsAgainst")),
            "goal_diff": _safe_int(raw.get("intGoalDifference")),
            "points": _safe_int(raw.get("intPoints")),
            "form": str(raw.get("strForm", "")),
            "source": "thesportsdb",
        }


def _safe_int(val: Any) -> int:
    """Safely convert to int, defaulting to 0."""
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0
