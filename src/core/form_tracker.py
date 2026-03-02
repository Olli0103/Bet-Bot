"""Redis-backed sliding window of last 5 results per team for form tracking."""
from __future__ import annotations

import json
import re
from typing import Tuple

from src.data.redis_cache import cache

FORM_WINDOW = 5
FORM_TTL = 7 * 24 * 3600  # 7 days


def _normalize_team(team: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (team or "").lower())


def _cache_key(team: str) -> str:
    return f"form:l5:{_normalize_team(team)}"


def get_form_l5(team: str) -> Tuple[float, int]:
    """Return (win_rate, games_played) for the last 5 results of a team."""
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
    """Append a result (won=True/False) to the team's sliding window."""
    k = _cache_key(team)
    data = cache.get_json(k)
    if not data or not isinstance(data, dict):
        data = {"games": []}
    games = data.get("games", [])
    games.append(1 if won else 0)
    # Keep only the last FORM_WINDOW results
    data["games"] = games[-FORM_WINDOW:]
    cache.set_json(k, data, ttl_seconds=FORM_TTL)
