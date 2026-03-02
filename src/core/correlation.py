"""Dynamic pairwise correlation penalties for combo bets.

Same-league legs are far more correlated than cross-sport legs (e.g., an EPL
"high-scoring week" trend affects all EPL matches).  Same-event, different-market
legs (e.g., Team A ML + Over 2.5 in the same game) have the highest correlation.
"""
from __future__ import annotations

from typing import Dict, List


# ---- penalty constants -------------------------------------------------------

SAME_EVENT_DIFF_MARKET = 0.80   # e.g., Team A ML + Team A Over 2.5
SAME_LEAGUE_PENALTY = 0.92      # two EPL games
SAME_SPORT_PENALTY = 0.97       # same sport, different league (e.g., EPL + La Liga)
CROSS_SPORT = 1.00              # independent (soccer + basketball)

# ---- sport & league extraction -----------------------------------------------

_LEAGUE_MAP: Dict[str, str] = {}


def _extract_league(sport_key: str) -> str:
    """Return the league portion of a sport key (e.g., 'soccer_epl' -> 'epl')."""
    parts = sport_key.split("_", 1)
    return parts[1] if len(parts) > 1 else sport_key


def _extract_sport(sport_key: str) -> str:
    """Return the sport prefix (e.g., 'soccer_epl' -> 'soccer')."""
    return sport_key.split("_", 1)[0]


# ---- main engine --------------------------------------------------------------

class CorrelationEngine:
    """Computes dynamic pairwise correlation penalties for combo legs."""

    @staticmethod
    def _pair_penalty(leg_a: Dict, leg_b: Dict) -> float:
        """Return the correlation penalty for a pair of legs."""
        event_a = leg_a.get("event_id", "")
        event_b = leg_b.get("event_id", "")

        # Same event, different market (highest correlation)
        if event_a == event_b:
            return SAME_EVENT_DIFF_MARKET

        sport_a = _extract_sport(leg_a.get("sport", ""))
        sport_b = _extract_sport(leg_b.get("sport", ""))

        # Cross-sport: independent
        if sport_a != sport_b:
            return CROSS_SPORT

        league_a = _extract_league(leg_a.get("sport", ""))
        league_b = _extract_league(leg_b.get("sport", ""))

        # Same league: high correlation
        if league_a == league_b:
            return SAME_LEAGUE_PENALTY

        # Same sport, different league: mild correlation
        return SAME_SPORT_PENALTY

    @classmethod
    def compute_combo_correlation(cls, legs: List[Dict]) -> float:
        """Pairwise correlation accumulation across all leg combinations.

        Returns a multiplier in (0, 1] that should be applied to the
        independent combined probability.
        """
        penalty = 1.0
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                penalty *= cls._pair_penalty(legs[i], legs[j])
        return penalty
