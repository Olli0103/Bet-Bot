"""Redis-backed Elo power rating system for team strength estimation."""
from __future__ import annotations

from typing import Dict

from src.core.sport_mapping import normalize_team as _normalize_team
from src.data.redis_cache import cache

DEFAULT_RATING = 1500.0
DEFAULT_K = 32
ELO_TTL = 90 * 24 * 3600  # 90 days

SPORT_K: Dict[str, int] = {
    "soccer": 24,
    "basketball": 20,
    "tennis": 32,
}


def _cache_key(team: str) -> str:
    return f"elo:rating:{_normalize_team(team)}"


class EloSystem:
    """Elo rating tracker with Redis persistence and sport-specific K-factors."""

    def __init__(self, k_factor: int = DEFAULT_K) -> None:
        self.k_factor = k_factor

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def get_rating(self, team: str) -> float:
        """Return the current Elo rating for *team*, defaulting to 1500."""
        k = _cache_key(team)
        data = cache.get_json(k)
        if data is None:
            return DEFAULT_RATING
        return float(data)

    def _set_rating(self, team: str, rating: float) -> None:
        k = _cache_key(team)
        cache.set_json(k, round(rating, 2), ttl_seconds=ELO_TTL)

    # ------------------------------------------------------------------
    # Elo mathematics
    # ------------------------------------------------------------------

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Standard Elo expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    # ------------------------------------------------------------------
    # Rating updates
    # ------------------------------------------------------------------

    def update(
        self,
        home: str,
        away: str,
        home_won: bool,
        sport: str = "",
    ) -> None:
        """Update Elo ratings for both teams after a match result."""
        k = SPORT_K.get(sport.lower(), self.k_factor) if sport else self.k_factor

        r_home = self.get_rating(home)
        r_away = self.get_rating(away)

        exp_home = self.expected_score(r_home, r_away)
        exp_away = 1.0 - exp_home

        actual_home = 1.0 if home_won else 0.0
        actual_away = 1.0 - actual_home

        new_home = r_home + k * (actual_home - exp_home)
        new_away = r_away + k * (actual_away - exp_away)

        self._set_rating(home, new_home)
        self._set_rating(away, new_away)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def get_elo_features(self, home: str, away: str) -> Dict[str, float]:
        """Return Elo-derived features for a home vs away matchup."""
        r_home = self.get_rating(home)
        r_away = self.get_rating(away)
        return {
            "elo_diff": round(r_home - r_away, 2),
            "elo_expected": round(self.expected_score(r_home, r_away), 4),
        }
