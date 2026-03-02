"""Historical odds volatility tracking and steam move detection."""
from __future__ import annotations

import math
from typing import Dict, List

from src.data.redis_cache import cache

VOLATILITY_TTL = 24 * 3600  # 24 hours


def _cache_key(team: str) -> str:
    normalized = "".join(c for c in team.lower() if c.isalnum())
    return f"volatility:{normalized}"


def record_odds_snapshot(team: str, implied_prob: float) -> None:
    """Record an implied probability snapshot for volatility calculation."""
    k = _cache_key(team)
    data = cache.get_json(k) or {"snapshots": []}
    snapshots = data.get("snapshots", [])
    snapshots.append(implied_prob)
    # Keep last 50 snapshots
    data["snapshots"] = snapshots[-50:]
    cache.set_json(k, data, ttl_seconds=VOLATILITY_TTL)


def get_volatility(team: str) -> float:
    """Return historical odds volatility (std dev of implied prob changes)."""
    k = _cache_key(team)
    data = cache.get_json(k) or {}
    snapshots = data.get("snapshots", [])
    if len(snapshots) < 3:
        return 0.0
    # Compute std dev of consecutive differences
    diffs = [snapshots[i] - snapshots[i - 1] for i in range(1, len(snapshots))]
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return math.sqrt(variance)


def detect_steam_move(
    current_odds: float,
    opening_odds: float,
    team_volatility: float,
    threshold_multiplier: float = 2.0,
) -> bool:
    """Detect if current movement exceeds threshold * normal volatility."""
    if opening_odds <= 1.0 or current_odds <= 1.0:
        return False
    current_ip = 1.0 / current_odds
    opening_ip = 1.0 / opening_odds
    movement = abs(current_ip - opening_ip)
    if team_volatility <= 0.001:
        # No historical data; flag large movements (>5% implied prob shift)
        return movement > 0.05
    return movement > (team_volatility * threshold_multiplier)


def get_volatility_features(home: str, away: str) -> Dict[str, float]:
    """Return volatility-related features for a match."""
    h_vol = get_volatility(home)
    a_vol = get_volatility(away)
    return {
        "home_volatility": h_vol,
        "away_volatility": a_vol,
    }
