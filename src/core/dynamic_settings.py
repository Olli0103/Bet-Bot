"""Redis-backed dynamic settings for Telegram toggle dashboard.

All settings are persisted in Redis so they survive restarts and can be
toggled via inline keyboard buttons in Telegram.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.core.sport_mapping import SPORT_DISPLAY_NAMES
from src.data.redis_cache import cache

log = logging.getLogger(__name__)

REDIS_KEY = "dynamic_settings:v1"


# Available sports: sourced from the central SPORT_MAPPING.
# Only expose leagues that Odds API supports for live odds.
_LIVE_SPORTS_KEYS = [
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_epl",
    "soccer_england_championship",
    "soccer_spain_la_liga",
    "soccer_spain_segunda_division",
    "soccer_italy_serie_a",
    "soccer_italy_serie_b",
    "soccer_france_ligue_one",
    "soccer_france_ligue_two",
    "soccer_uefa_champs_league",
    "basketball_nba",
    "basketball_euroleague",
    "americanfootball_nfl",
    "icehockey_nhl",
    "tennis_atp",
    "tennis_wta",
]

AVAILABLE_SPORTS: Dict[str, str] = {
    k: SPORT_DISPLAY_NAMES.get(k, k) for k in _LIVE_SPORTS_KEYS
}

AVAILABLE_MARKETS: Dict[str, str] = {
    "h2h": "H2H",
    "totals": "Totals",
    "spreads": "Spreads",
    "double_chance": "Double Chance",
    "draw_no_bet": "Draw No Bet",
}

AVAILABLE_COMBO_SIZES: List[int] = [10, 20, 30]


class DynamicSettingsManager:
    """Redis-backed dynamic settings, togglable via Telegram.

    Supports per-owner settings via owner_chat_id. When set, settings
    are isolated per portfolio owner. Falls back to global defaults.
    """

    DEFAULTS: Dict[str, Any] = {
        "active_sports": [
            "soccer_germany_bundesliga",
            "soccer_epl",
            "basketball_nba",
            "icehockey_nhl",
        ],
        "active_markets": ["h2h", "totals", "spreads", "double_chance", "draw_no_bet"],
        "min_odds_threshold": 1.20,
        "target_combo_sizes": [10, 20, 30],
    }

    def __init__(self, owner_chat_id: str = ""):
        self._owner = owner_chat_id

    def _redis_key(self) -> str:
        """Return owner-scoped Redis key if owner is set, else global."""
        if self._owner:
            return f"{REDIS_KEY}:owner:{self._owner}"
        return REDIS_KEY

    def get_all(self) -> Dict[str, Any]:
        """Return all settings, falling back to defaults if Redis is empty."""
        data = cache.get_json(self._redis_key())
        if data and isinstance(data, dict):
            merged = dict(self.DEFAULTS)
            merged.update(data)
            return merged
        # If owner-scoped and not found, fall back to global
        if self._owner:
            global_data = cache.get_json(REDIS_KEY)
            if global_data and isinstance(global_data, dict):
                merged = dict(self.DEFAULTS)
                merged.update(global_data)
                return merged
        return dict(self.DEFAULTS)

    def _save(self, data: Dict[str, Any]) -> None:
        cache.set_json(self._redis_key(), data, ttl_seconds=365 * 24 * 3600)

    def get(self, key: str) -> Any:
        return self.get_all().get(key, self.DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> None:
        data = self.get_all()
        data[key] = value
        self._save(data)

    # --- Toggles ---

    def toggle_sport(self, sport_key: str) -> bool:
        """Toggle a sport on/off. Returns new state (True=active)."""
        data = self.get_all()
        active: List[str] = data.get("active_sports", [])
        if sport_key in active:
            active.remove(sport_key)
            new_state = False
        else:
            active.append(sport_key)
            new_state = True
        data["active_sports"] = active
        self._save(data)
        return new_state

    def toggle_market(self, market_key: str) -> bool:
        """Toggle a market on/off. Returns new state (True=active)."""
        data = self.get_all()
        active: List[str] = data.get("active_markets", [])
        if market_key in active:
            active.remove(market_key)
            new_state = False
        else:
            active.append(market_key)
            new_state = True
        data["active_markets"] = active
        self._save(data)
        return new_state

    def toggle_combo_size(self, size: int) -> bool:
        """Toggle a combo size on/off. Returns new state (True=active)."""
        data = self.get_all()
        sizes: List[int] = data.get("target_combo_sizes", [])
        if size in sizes:
            sizes.remove(size)
            new_state = False
        else:
            sizes.append(size)
            sizes.sort()
            new_state = True
        data["target_combo_sizes"] = sizes
        self._save(data)
        return new_state

    def set_min_odds(self, value: float) -> None:
        """Set minimum odds threshold."""
        data = self.get_all()
        data["min_odds_threshold"] = round(max(1.01, value), 2)
        self._save(data)

    # --- Convenience ---

    def get_active_sports(self) -> List[str]:
        return self.get("active_sports") or []

    def get_active_markets(self) -> List[str]:
        return self.get("active_markets") or []

    def get_min_odds(self) -> float:
        return float(self.get("min_odds_threshold") or 1.20)

    def get_combo_sizes(self) -> List[int]:
        return self.get("target_combo_sizes") or [10, 20, 30]


# Singleton
dynamic_settings = DynamicSettingsManager()
