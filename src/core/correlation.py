"""Dynamic pairwise correlation penalties for combo bets.

Same-league legs are far more correlated than cross-sport legs (e.g., an EPL
"high-scoring week" trend affects all EPL matches).  Same-event, different-market
legs (SGPs) have *directional* correlation that can be positive (boosting the
true probability) or negative (penalizing it).

Key insight: blindly penalizing all SGP legs is mathematically wrong.
"Favorite Win + Over 2.5 Goals" are *positively* correlated — the true joint
probability is *higher* than the product of marginals.  Only negatively
correlated combinations (e.g., "Favorite Win + Under 0.5 Goals") should be
penalized.
"""
from __future__ import annotations

import logging
from typing import Dict, List

log = logging.getLogger(__name__)


# ---- penalty constants -------------------------------------------------------

# Cross-event penalties (unchanged)
SAME_LEAGUE_PENALTY = 0.92      # two games from same league
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


# ---- SGP directional correlation ---------------------------------------------

def _classify_market(leg: Dict) -> str:
    """Classify a leg into a market category for correlation lookup."""
    market = leg.get("market", leg.get("market_type", "")).lower()
    if "total" in market or "over_under" in market:
        return "totals"
    if "spread" in market:
        return "spreads"
    if "btts" in market or "both_teams" in market:
        return "btts"
    return "h2h"


def _is_over(leg: Dict) -> bool:
    """Check if the selection is an 'over' bet."""
    return "over" in leg.get("selection", "").lower()


def _is_favorite(leg: Dict) -> bool:
    """Heuristic: odds < 2.0 implies favorite."""
    return leg.get("odds", 99.0) < 2.0


def _same_event_pair_multiplier(leg_a: Dict, leg_b: Dict) -> float:
    """Compute directional correlation multiplier for a same-game pair.

    Positive correlation (multiplier > 1.0) boosts joint probability.
    Negative correlation (multiplier < 1.0) penalizes joint probability.

    Market pair mappings (empirical, based on soccer correlation matrices):
    - Favorite H2H + Over:    +0.15 (goals come from dominant teams)
    - Underdog H2H + Under:   +0.10 (underdog wins tend to be low-scoring)
    - Favorite H2H + Under:   -0.30 (contradictory: favorite wins usually high-scoring)
    - Underdog H2H + Over:    -0.15 (less contradictory but still unusual)
    - H2H + BTTS:             +0.05 (mild positive for balanced games)
    - Same market type:        0.80 (e.g., two goalscorers — strong positive)
    """
    cat_a = _classify_market(leg_a)
    cat_b = _classify_market(leg_b)
    markets = {cat_a, cat_b}

    # Same market type in same event (e.g., two goalscorer bets) — strong dependency
    if cat_a == cat_b:
        return 0.80

    # H2H + Totals (over/under) — direction-dependent
    if "h2h" in markets and "totals" in markets:
        h2h_leg = leg_a if cat_a == "h2h" else leg_b
        totals_leg = leg_a if cat_a == "totals" else leg_b

        fav = _is_favorite(h2h_leg)
        over = _is_over(totals_leg)

        if fav and over:
            return 1.15   # Positive: favorite dominance = more goals
        if not fav and not over:
            return 1.10   # Positive: underdog grinds = low scoring
        if fav and not over:
            return 0.70   # Negative: favorite winning with few goals is rare
        # not fav and over
        return 0.85       # Mild negative: underdog winning high-scoring is unusual

    # H2H + BTTS
    if "h2h" in markets and "btts" in markets:
        return 1.05  # Mild positive correlation

    # H2H + Spreads — nearly redundant markets
    if "h2h" in markets and "spreads" in markets:
        return 0.85  # Strong dependency, penalize duplication

    # Totals + BTTS — positive correlation (more goals = both teams score)
    if "totals" in markets and "btts" in markets:
        totals_leg = leg_a if cat_a == "totals" else leg_b
        if _is_over(totals_leg):
            return 1.10  # Over + BTTS = positive
        return 0.85      # Under + BTTS = contradictory

    # Default: mild penalty for unknown same-event combos
    return 0.90


# ---- main engine --------------------------------------------------------------

class CorrelationEngine:
    """Computes dynamic pairwise correlation penalties for combo legs.

    For same-event pairs (SGPs), uses directional correlation that can
    *boost* positively correlated legs (multiplier > 1.0) instead of
    blindly penalizing all same-event pairs.
    """

    @staticmethod
    def _pair_penalty(leg_a: Dict, leg_b: Dict) -> float:
        """Return the correlation multiplier for a pair of legs."""
        event_a = leg_a.get("event_id", "")
        event_b = leg_b.get("event_id", "")

        # Same event: use directional correlation
        if event_a == event_b:
            mult = _same_event_pair_multiplier(leg_a, leg_b)
            log.debug(
                "SGP pair %s/%s: markets=%s/%s multiplier=%.2f",
                leg_a.get("selection", "?"), leg_b.get("selection", "?"),
                _classify_market(leg_a), _classify_market(leg_b), mult,
            )
            return mult

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

        Returns a multiplier that should be applied to the independent
        combined probability.  The multiplier can be > 1.0 when
        positively correlated SGP legs dominate the combo.

        Final result is clamped to [0.50, 2.50] to prevent extreme values.
        """
        multiplier = 1.0
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                multiplier *= cls._pair_penalty(legs[i], legs[j])
        return max(0.50, min(2.50, multiplier))
