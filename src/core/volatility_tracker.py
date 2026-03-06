"""Historical odds volatility tracking and steam move detection."""
from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

from src.data.redis_cache import cache

VOLATILITY_TTL = 24 * 3600  # 24 hours


def _cache_key(team: str) -> str:
    normalized = "".join(c for c in team.lower() if c.isalnum())
    return f"volatility:{normalized}"


def record_odds_snapshot(team: str, implied_prob: float) -> None:
    """Record an implied probability snapshot.

    Keep legacy float-only storage for test/backward compatibility.
    """
    k = _cache_key(team)
    data = cache.get_json(k) or {"snapshots": []}
    snapshots = data.get("snapshots", [])
    snapshots.append(float(implied_prob))
    # Keep last 50 snapshots
    data["snapshots"] = snapshots[-50:]
    cache.set_json(k, data, ttl_seconds=VOLATILITY_TTL)


def _extract_probs(snapshots: list) -> List[float]:
    """Extract implied probabilities, supporting both old (float) and new (dict) formats."""
    out: List[float] = []
    for s in snapshots:
        if isinstance(s, dict):
            out.append(float(s.get("ip", 0.0)))
        elif isinstance(s, (int, float)):
            out.append(float(s))
    return out


def get_volatility(team: str) -> float:
    """Return historical odds volatility (std dev of implied prob changes)."""
    k = _cache_key(team)
    data = cache.get_json(k) or {}
    probs = _extract_probs(data.get("snapshots", []))
    if len(probs) < 3:
        return 0.0
    diffs = [probs[i] - probs[i - 1] for i in range(1, len(probs))]
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


def get_line_velocity(team: str) -> float:
    """Line movement velocity: implied prob change per hour over the last window.

    Tracks how *fast* the line is moving, not just the magnitude.
    A 5% shift in 15 minutes (velocity=20%/h) is a much stronger sharp
    signal than 5% over 12 hours (velocity=0.4%/h).

    Returns 0.0 when fewer than 2 timestamped snapshots are available.
    """
    k = _cache_key(team)
    data = cache.get_json(k) or {}
    snapshots = data.get("snapshots", [])

    # Need at least 2 timestamped snapshots
    ts_snaps: List[Tuple[float, float]] = []
    for s in snapshots:
        if isinstance(s, dict) and "ts" in s and "ip" in s:
            ts_snaps.append((float(s["ts"]), float(s["ip"])))
    if len(ts_snaps) < 2:
        return 0.0

    # Use earliest and latest snapshots for velocity over the full window
    first_ts, first_ip = ts_snaps[0]
    last_ts, last_ip = ts_snaps[-1]
    elapsed_hours = (last_ts - first_ts) / 3600.0
    if elapsed_hours < 0.01:  # < 36 seconds — avoid division noise
        return 0.0

    return (last_ip - first_ip) / elapsed_hours


def get_volatility_features(home: str, away: str) -> Dict[str, float]:
    """Return volatility-related features for a match."""
    h_vol = get_volatility(home)
    a_vol = get_volatility(away)
    return {
        "home_volatility": h_vol,
        "away_volatility": a_vol,
    }
