"""Head-to-head historical stats from PlacedBet records."""
from __future__ import annotations

import re
from typing import Dict

from sqlalchemy import select

from src.data.models import PlacedBet
from src.data.postgres import SessionLocal
from src.data.redis_cache import cache

H2H_TTL = 24 * 3600  # 24 hours


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _cache_key(home: str, away: str) -> str:
    return f"h2h:{_normalize(home)}:{_normalize(away)}"


def get_h2h_features(home: str, away: str) -> Dict[str, float]:
    """Return head-to-head stats from historical placed bets."""
    k = _cache_key(home, away)
    cached = cache.get_json(k)
    if cached is not None:
        return cached

    norm_home = _normalize(home)
    norm_away = _normalize(away)

    h2h_home_wins = 0
    h2h_away_wins = 0
    h2h_draws = 0

    try:
        with SessionLocal() as db:
            bets = db.scalars(select(PlacedBet).where(PlacedBet.status.in_(["won", "lost"]))).all()

        for bet in bets:
            sel = (bet.selection or "").strip()
            # Check if this bet involves both teams (selection contains both names)
            sel_lower = sel.lower()
            if norm_home not in _normalize(sel_lower) and norm_away not in _normalize(sel_lower):
                continue
            # Check event context for this matchup
            if "vs" in sel:
                parts = sel.split("vs")
                if len(parts) == 2:
                    left = _normalize(parts[0])
                    right = _normalize(parts[1])
                    if not ((norm_home in left and norm_away in right) or
                            (norm_away in left and norm_home in right)):
                        continue
            else:
                continue

            pick = sel.split("(")[0].strip() if "(" in sel else sel
            if bet.status == "won":
                if _normalize(pick) == norm_home or norm_home in _normalize(pick):
                    h2h_home_wins += 1
                elif _normalize(pick) == norm_away or norm_away in _normalize(pick):
                    h2h_away_wins += 1
                elif "draw" in _normalize(pick):
                    h2h_draws += 1
    except Exception:
        pass

    total = h2h_home_wins + h2h_away_wins + h2h_draws
    features = {
        "h2h_home_wins": float(h2h_home_wins),
        "h2h_away_wins": float(h2h_away_wins),
        "h2h_draws": float(h2h_draws),
        "h2h_total": float(total),
        "h2h_home_winrate": round(h2h_home_wins / max(1, total), 4),
    }
    cache.set_json(k, features, ttl_seconds=H2H_TTL)
    return features
