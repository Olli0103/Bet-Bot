"""Redis-backed dynamic settings for Telegram toggle dashboard.

All settings are persisted in Redis so they survive restarts and can be
toggled via inline keyboard buttons in Telegram.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

REDIS_KEY = "dynamic_settings:v1"


# Available sports (OddsAPI keys)
AVAILABLE_SPORTS: Dict[str, str] = {
    "soccer_germany_bundesliga": "Bundesliga",
    "soccer_epl": "EPL",
    "soccer_spain_la_liga": "La Liga",
    "soccer_italy_serie_a": "Serie A",
    "soccer_france_ligue_one": "Ligue 1",
    "soccer_uefa_champs_league": "CL",
    "basketball_nba": "NBA",
    "basketball_euroleague": "EuroLeague",
    "americanfootball_nfl": "NFL",
    "icehockey_nhl": "NHL",
    "tennis_atp": "ATP",
    "tennis_wta": "WTA",
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
    """Redis-backed dynamic settings, togglable via Telegram."""

    DEFAULTS: Dict[str, Any] = {
        "active_sports": [
            "soccer_germany_bundesliga",
            "soccer_epl",
            "basketball_nba",
            "tennis_atp",
        ],
        "active_markets": ["h2h", "totals", "spreads", "double_chance", "draw_no_bet"],
        "min_odds_threshold": 1.20,
        "target_combo_sizes": [10, 20, 30],
    }

    def get_all(self) -> Dict[str, Any]:
        """Return all settings, falling back to defaults if Redis is empty."""
        data = cache.get_json(REDIS_KEY)
        if data and isinstance(data, dict):
            merged = dict(self.DEFAULTS)
            merged.update(data)
            return merged
        return dict(self.DEFAULTS)

    def _save(self, data: Dict[str, Any]) -> None:
        cache.set_json(REDIS_KEY, data, ttl_seconds=365 * 24 * 3600)

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
