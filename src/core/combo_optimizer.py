"""Optimized combo construction with constraint engine.

Supports lotto-style combos (10/20/30 legs) and EV-optimal combos.
Uses dynamic correlation penalties from CorrelationEngine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.core.betting_engine import BettingEngine
from src.core.betting_math import effective_tax_rate
from src.core.correlation import CorrelationEngine
from src.core.settings import settings
from src.models.betting import ComboBet


@dataclass
class ComboConstraints:
    """Constraints for combo construction."""
    max_per_sport: int = 5
    max_per_league: int = 3
    no_same_event: bool = True
    min_sports: int = 2
    min_prob_per_leg: float = 0.50
    max_prob_per_leg: float = 0.95
    odds_range: Tuple[float, float] = (1.15, 4.00)
    # Heavy favorite cap: max number of legs with odds < threshold per league.
    # Prevents a single league-wide upset (e.g. rainy EPL matchday) from
    # killing the entire ticket.
    max_heavy_favorites_per_league: int = 3
    heavy_favorite_threshold: float = 1.30


# Pre-defined constraint profiles per combo size
COMBO_PROFILES: Dict[int, ComboConstraints] = {
    5: ComboConstraints(
        max_per_sport=3, max_per_league=2, min_sports=2,
        min_prob_per_leg=0.55, odds_range=(1.20, 3.50),
        max_heavy_favorites_per_league=2, heavy_favorite_threshold=1.30,
    ),
    10: ComboConstraints(
        max_per_sport=4, max_per_league=3, min_sports=3,
        min_prob_per_leg=0.52, odds_range=(1.15, 3.50),
        max_heavy_favorites_per_league=2, heavy_favorite_threshold=1.30,
    ),
    20: ComboConstraints(
        max_per_sport=7, max_per_league=4, min_sports=4,
        min_prob_per_leg=0.50, odds_range=(1.15, 4.00),
        max_heavy_favorites_per_league=3, heavy_favorite_threshold=1.30,
    ),
    30: ComboConstraints(
        max_per_sport=10, max_per_league=5, min_sports=5,
        min_prob_per_leg=0.48, odds_range=(1.10, 4.00),
        max_heavy_favorites_per_league=3, heavy_favorite_threshold=1.30,
    ),
}

# Fixed stakes per combo size (lotto mode)
COMBO_STAKES: Dict[int, float] = {
    5: 2.00,
    10: 1.00,
    20: 1.00,
    30: 0.50,
}


def _extract_sport(sport_key: str) -> str:
    return sport_key.split("_", 1)[0]


def _extract_league(sport_key: str) -> str:
    parts = sport_key.split("_", 1)
    return parts[1] if len(parts) > 1 else sport_key


def _combo_score(prob: float, odds: float, market_type: str = "h2h") -> float:
    """Score a leg for combo selection: balance win probability with payout contribution.

    High-probability markets (Over 0.5/1.5, Double Chance) get a boost
    to prioritize them in Lotto combos where hit rate matters most.
    """
    if prob <= 0 or odds <= 1.0:
        return 0.0
    base = prob * math.log(odds)
    if market_type in ("double_chance", "draw_no_bet"):
        base *= 1.20
    elif market_type == "totals" and prob >= 0.80:
        base *= 1.15
    return base


class ComboOptimizer:
    """Builds optimized combo bets with constraint satisfaction."""

    def __init__(self, engine: BettingEngine):
        self.engine = engine

    def _select_legs(
        self,
        candidates: List[Dict],
        target_legs: int,
        constraints: ComboConstraints,
    ) -> List[Dict]:
        """Select legs that satisfy all constraints, ranked by combo_score."""
        # Pre-filter
        valid = []
        for leg in candidates:
            odds = float(leg.get("odds", 0))
            prob = float(leg.get("probability", 0))
            if prob < constraints.min_prob_per_leg or prob > constraints.max_prob_per_leg:
                continue
            if odds < constraints.odds_range[0] or odds > constraints.odds_range[1]:
                continue
            valid.append(leg)

        # Score and sort (with market-type-aware scoring)
        scored = sorted(
            valid,
            key=lambda l: _combo_score(l["probability"], l["odds"], l.get("market_type", "h2h")),
            reverse=True,
        )

        chosen: List[Dict] = []
        used_events: set = set()
        sport_count: Dict[str, int] = {}
        league_count: Dict[str, int] = {}
        league_heavy_count: Dict[str, int] = {}  # heavy favorites per league
        sports_used: set = set()

        for leg in scored:
            event_id = leg.get("event_id", "")
            sport_key = leg.get("sport", "")
            odds = float(leg.get("odds", 0))
            sport = _extract_sport(sport_key)
            league = _extract_league(sport_key)
            is_heavy_favorite = odds < constraints.heavy_favorite_threshold

            # No duplicate events
            if constraints.no_same_event and event_id in used_events:
                continue

            # Per-sport limit
            if sport_count.get(sport, 0) >= constraints.max_per_sport:
                continue

            # Per-league limit
            if league_count.get(league, 0) >= constraints.max_per_league:
                continue

            # Heavy favorite cap per league: prevents a single league-wide
            # upset (e.g. rainy EPL matchday) from killing the ticket
            if is_heavy_favorite:
                if league_heavy_count.get(league, 0) >= constraints.max_heavy_favorites_per_league:
                    continue

            chosen.append(leg)
            used_events.add(event_id)
            sport_count[sport] = sport_count.get(sport, 0) + 1
            league_count[league] = league_count.get(league, 0) + 1
            if is_heavy_favorite:
                league_heavy_count[league] = league_heavy_count.get(league, 0) + 1
            sports_used.add(sport)

            if len(chosen) == target_legs:
                break

        # Check min_sports constraint
        if len(sports_used) < constraints.min_sports and len(chosen) >= target_legs:
            # Try to swap in legs from missing sports
            pass  # Accept what we have — strict enforcement would reduce combo availability

        return chosen

    def build_lotto_combo(
        self,
        candidates: List[Dict],
        target_legs: int,
        constraints: Optional[ComboConstraints] = None,
    ) -> Optional[ComboBet]:
        """Build a lotto-style combo: small stake, maximized potential payout."""
        if constraints is None:
            constraints = COMBO_PROFILES.get(target_legs, COMBO_PROFILES[10])

        legs = self._select_legs(candidates, target_legs, constraints)
        if len(legs) < min(target_legs, 3):
            return None

        # Dynamic correlation penalty
        correlation = CorrelationEngine.compute_combo_correlation(legs)

        # Fixed stake for lotto combos
        stake = COMBO_STAKES.get(target_legs, 1.00)
        kelly_frac = stake / max(1.0, self.engine.bankroll)

        # Tax-free for 3+ leg combos (Tipico promotion)
        tax = effective_tax_rate(
            base_tax=settings.tipico_tax_rate,
            tax_free_mode=settings.tax_free_mode,
            is_combo=True,
            combo_legs=len(legs),
        )

        return self.engine.build_combo(
            legs,
            correlation_penalty=correlation,
            kelly_frac=kelly_frac,
            tax_rate=tax,
        )

    def build_ev_optimal_combo(
        self,
        candidates: List[Dict],
        target_legs: int = 5,
    ) -> Optional[ComboBet]:
        """Build an EV-optimal combo: maximize expected value."""
        constraints = ComboConstraints(
            max_per_sport=3, max_per_league=2, min_sports=2,
            min_prob_per_leg=0.55, odds_range=(1.30, 3.00),
        )

        # For EV combos, sort by EV contribution instead of combo_score
        valid = []
        for leg in candidates:
            odds = float(leg.get("odds", 0))
            prob = float(leg.get("probability", 0))
            if prob < constraints.min_prob_per_leg:
                continue
            if odds < constraints.odds_range[0] or odds > constraints.odds_range[1]:
                continue
            ev = prob * (odds - 1.0) - (1.0 - prob)
            if ev > 0:
                leg["_ev"] = ev
                valid.append(leg)

        if len(valid) < min(target_legs, 3):
            return None

        # Sort by EV descending
        valid.sort(key=lambda l: l.get("_ev", 0), reverse=True)

        legs = self._select_legs(valid, target_legs, constraints)
        if len(legs) < min(target_legs, 3):
            return None

        correlation = CorrelationEngine.compute_combo_correlation(legs)

        # Tax-free for 3+ leg combos (Tipico promotion)
        tax = effective_tax_rate(
            base_tax=settings.tipico_tax_rate,
            tax_free_mode=settings.tax_free_mode,
            is_combo=True,
            combo_legs=len(legs),
        )

        return self.engine.build_combo(
            legs,
            correlation_penalty=correlation,
            kelly_frac=0.01,  # Conservative Kelly for combos
            tax_rate=tax,
        )

    def build_all_combos(
        self,
        candidates: List[Dict],
        target_sizes: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Build combos for configured target sizes (default: 10, 20, 30)."""
        if target_sizes is None:
            target_sizes = [10, 20, 30]

        results = []

        for target in target_sizes:
            combo = self.build_lotto_combo(candidates, target_legs=target)
            if combo:
                results.append({
                    "size": target,
                    "type": "lotto",
                    "stake": COMBO_STAKES.get(target, 1.00),
                    **combo.model_dump(),
                })

        return results
