"""Form tracking — last N results per team.

Primary source: TeamMatchStats (actual match results from TheSportsDB).
Fallback: Redis sliding window (updated by autograding from bet outcomes).

Using TeamMatchStats breaks the circular dependency where form was only
derived from our own placed bets.
"""
from __future__ import annotations

import logging
from typing import Tuple

from src.core.sport_mapping import normalize_team as _normalize_team
from src.data.redis_cache import cache

log = logging.getLogger(__name__)

FORM_WINDOW = 5
FORM_TTL = 7 * 24 * 3600  # 7 days


def _cache_key(team: str) -> str:
    return f"form:l5:{_normalize_team(team)}"


def _form_from_match_stats(team: str) -> Tuple[float, int]:
    """Query TeamMatchStats for the last FORM_WINDOW results.

    Returns (win_rate, games_played) or (None, 0) if no data.
    """
    try:
        from sqlalchemy import select
        from src.data.models import TeamMatchStats
        from src.data.postgres import SessionLocal

        norm = _normalize_team(team)

        with SessionLocal() as db:
            rows = db.scalars(
                select(TeamMatchStats)
                .where(TeamMatchStats.team == norm)
                .order_by(TeamMatchStats.match_date.desc())
                .limit(FORM_WINDOW)
            ).all()

            if not rows:
                # Try un-normalized name (TheSportsDB stores display names)
                rows = db.scalars(
                    select(TeamMatchStats)
                    .where(TeamMatchStats.team == team)
                    .order_by(TeamMatchStats.match_date.desc())
                    .limit(FORM_WINDOW)
                ).all()

        if not rows:
            return 0.5, 0

        wins = sum(1 for r in rows if r.result == "W")
        total = len(rows)
        return round(wins / max(1, total), 4), total
    except Exception as exc:
        log.debug("TeamMatchStats form lookup failed for %s: %s", team, exc)
        return 0.5, 0


def get_form_l5(team: str) -> Tuple[float, int]:
    """Return (win_rate, games_played) for the last 5 results of a team.

    Tries TeamMatchStats first (ground truth), falls back to Redis cache
    (populated by autograding from bet outcomes).
    """
    # Primary: TeamMatchStats (actual match data)
    wr, gp = _form_from_match_stats(team)
    if gp > 0:
        return wr, gp

    # Fallback: Redis sliding window (from autograding)
    k = _cache_key(team)
    data = cache.get_json(k)
    if not data or not isinstance(data, dict):
        return 0.5, 0
    games = data.get("games", [])
    if not games:
        return 0.5, 0
    wins = sum(1 for g in games if g == 1)
    total = len(games)
    return round(wins / max(1, total), 4), total


def update_form(team: str, won: bool) -> None:
    """Append a result (won=True/False) to the Redis sliding window.

    This is the fallback data source; TeamMatchStats is the primary.
    """
    k = _cache_key(team)
    data = cache.get_json(k)
    if not data or not isinstance(data, dict):
        data = {"games": []}
    games = data.get("games", [])
    games.append(1 if won else 0)
    # Keep only the last FORM_WINDOW results
    data["games"] = games[-FORM_WINDOW:]
    cache.set_json(k, data, ttl_seconds=FORM_TTL)
