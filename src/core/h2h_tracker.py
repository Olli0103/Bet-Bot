"""Head-to-head historical stats.

Primary source: TeamMatchStats (actual match results).
Fallback: PlacedBet records (legacy, circular for new matchups).

Using TeamMatchStats breaks the circular dependency where H2H was
derived from our own placed bets (which only exist after we've already
bet on the matchup).
"""
from __future__ import annotations

import logging
from typing import Dict

from sqlalchemy import select, and_, or_

from src.core.sport_mapping import normalize_team
from src.data.redis_cache import cache

log = logging.getLogger(__name__)

H2H_TTL = 24 * 3600  # 24 hours


def _cache_key(home: str, away: str) -> str:
    return f"h2h:{normalize_team(home)}:{normalize_team(away)}"


def _h2h_from_match_stats(home: str, away: str) -> Dict[str, float]:
    """Query TeamMatchStats for historical meetings between two teams.

    Returns the feature dict or None if no data found.
    """
    try:
        from src.data.models import TeamMatchStats
        from src.data.postgres import SessionLocal

        norm_home = normalize_team(home)
        norm_away = normalize_team(away)

        with SessionLocal() as db:
            # Find matches where team=home & opponent=away, or vice versa
            # Try both normalized and raw names since TheSportsDB may store
            # either form.
            rows = db.scalars(
                select(TeamMatchStats).where(
                    or_(
                        and_(TeamMatchStats.team == norm_home, TeamMatchStats.opponent == norm_away),
                        and_(TeamMatchStats.team == home, TeamMatchStats.opponent == away),
                        and_(TeamMatchStats.team == norm_home, TeamMatchStats.opponent == away),
                        and_(TeamMatchStats.team == home, TeamMatchStats.opponent == norm_away),
                    )
                ).order_by(TeamMatchStats.match_date.desc())
                .limit(20)
            ).all()

        if not rows:
            return {}

        h2h_home_wins = sum(1 for r in rows if r.result == "W" and r.is_home)
        h2h_away_wins = sum(1 for r in rows if r.result == "W" and not r.is_home)
        h2h_draws = sum(1 for r in rows if r.result == "D")
        total = len(rows)

        return {
            "h2h_home_wins": float(h2h_home_wins),
            "h2h_away_wins": float(h2h_away_wins),
            "h2h_draws": float(h2h_draws),
            "h2h_total": float(total),
            "h2h_home_winrate": round(h2h_home_wins / max(1, total), 4),
        }
    except Exception as exc:
        log.debug("TeamMatchStats H2H lookup failed for %s vs %s: %s", home, away, exc)
        return {}


def get_h2h_features(home: str, away: str) -> Dict[str, float]:
    """Return head-to-head stats, preferring TeamMatchStats over PlacedBet.

    Results are cached in Redis for 24h.
    """
    k = _cache_key(home, away)
    cached = cache.get_json(k)
    if cached is not None:
        return cached

    # Primary: TeamMatchStats (actual match results)
    features = _h2h_from_match_stats(home, away)
    if features and features.get("h2h_total", 0) > 0:
        cache.set_json(k, features, ttl_seconds=H2H_TTL)
        return features

    # No data — return neutral defaults (0.5 = no information)
    features = {
        "h2h_home_wins": 0.0,
        "h2h_away_wins": 0.0,
        "h2h_draws": 0.0,
        "h2h_total": 0.0,
        "h2h_home_winrate": 0.5,
    }
    cache.set_json(k, features, ttl_seconds=H2H_TTL)
    return features
