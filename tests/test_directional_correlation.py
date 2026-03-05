"""Tests for directional SGP correlation logic.

Verifies that positively correlated same-game legs (e.g., Favorite Win +
Over 2.5) get a multiplier > 1.0 (boosting probability), while negatively
correlated legs (e.g., Favorite Win + Under 0.5) get a multiplier < 1.0.
"""
import pytest

from src.core.correlation import (
    CorrelationEngine,
    _same_event_pair_multiplier,
    _classify_market,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leg(event_id="e1", selection="Home", odds=1.80, sport="soccer_epl",
         market="h2h", probability=0.60):
    return {
        "event_id": event_id,
        "selection": selection,
        "odds": odds,
        "probability": probability,
        "sport": sport,
        "market": market,
        "market_type": market,
        "home_team": "Team A",
        "away_team": "Team B",
    }


# ---------------------------------------------------------------------------
# Market classification
# ---------------------------------------------------------------------------

class TestClassifyMarket:
    def test_h2h(self):
        assert _classify_market({"market": "h2h"}) == "h2h"

    def test_totals(self):
        assert _classify_market({"market": "totals 2.5"}) == "totals"
        assert _classify_market({"market": "over_under"}) == "totals"

    def test_spreads(self):
        assert _classify_market({"market": "spreads +3.5"}) == "spreads"

    def test_btts(self):
        assert _classify_market({"market": "btts"}) == "btts"
        assert _classify_market({"market": "both_teams_to_score"}) == "btts"


# ---------------------------------------------------------------------------
# Directional correlation: Favorite Win + Over = positive correlation
# ---------------------------------------------------------------------------

class TestDirectionalSGP:
    def test_favorite_win_plus_over_boosts(self):
        """Favorite Win + Over 2.5 should have multiplier > 1.0."""
        h2h_leg = _leg(selection="Home", odds=1.50, market="h2h")
        over_leg = _leg(selection="Over 2.5", odds=1.90, market="totals 2.5")
        mult = _same_event_pair_multiplier(h2h_leg, over_leg)
        assert mult > 1.0, f"Expected boost (>1.0), got {mult}"
        assert mult == pytest.approx(1.15)

    def test_underdog_win_plus_under_boosts(self):
        """Underdog Win + Under 2.5 should have multiplier > 1.0."""
        h2h_leg = _leg(selection="Away", odds=3.50, market="h2h")
        under_leg = _leg(selection="Under 2.5", odds=1.90, market="totals 2.5")
        mult = _same_event_pair_multiplier(h2h_leg, under_leg)
        assert mult > 1.0, f"Expected boost (>1.0), got {mult}"
        assert mult == pytest.approx(1.10)

    def test_favorite_win_plus_under_penalizes(self):
        """Favorite Win + Under 0.5 should have multiplier < 1.0 (contradictory)."""
        h2h_leg = _leg(selection="Home", odds=1.50, market="h2h")
        under_leg = _leg(selection="Under 0.5", odds=4.00, market="totals 0.5")
        mult = _same_event_pair_multiplier(h2h_leg, under_leg)
        assert mult < 1.0, f"Expected penalty (<1.0), got {mult}"
        assert mult == pytest.approx(0.70)

    def test_underdog_plus_over_mild_penalty(self):
        """Underdog Win + Over 2.5 should get mild negative correlation."""
        h2h_leg = _leg(selection="Away", odds=3.50, market="h2h")
        over_leg = _leg(selection="Over 2.5", odds=1.90, market="totals 2.5")
        mult = _same_event_pair_multiplier(h2h_leg, over_leg)
        assert mult < 1.0

    def test_same_market_strong_dependency(self):
        """Two goalscorer bets in the same game: strong positive dependency."""
        leg_a = _leg(selection="Player A", market="goalscorer")
        leg_b = _leg(selection="Player B", market="goalscorer")
        mult = _same_event_pair_multiplier(leg_a, leg_b)
        assert mult == pytest.approx(0.80)

    def test_h2h_plus_btts_mild_positive(self):
        """H2H + BTTS should have mild positive correlation."""
        h2h_leg = _leg(selection="Home", odds=1.50, market="h2h")
        btts_leg = _leg(selection="Yes", market="btts")
        mult = _same_event_pair_multiplier(h2h_leg, btts_leg)
        assert mult == pytest.approx(1.05)

    def test_over_plus_btts_positive(self):
        """Over + BTTS = positive (more goals = both teams score)."""
        over_leg = _leg(selection="Over 2.5", market="totals 2.5")
        btts_leg = _leg(selection="Yes", market="btts")
        mult = _same_event_pair_multiplier(over_leg, btts_leg)
        assert mult > 1.0

    def test_under_plus_btts_negative(self):
        """Under + BTTS = contradictory."""
        under_leg = _leg(selection="Under 2.5", market="totals 2.5")
        btts_leg = _leg(selection="Yes", market="btts")
        mult = _same_event_pair_multiplier(under_leg, btts_leg)
        assert mult < 1.0


# ---------------------------------------------------------------------------
# Cross-event penalties unchanged
# ---------------------------------------------------------------------------

class TestCrossEventPenalties:
    def test_cross_sport_independent(self):
        """Soccer + Basketball legs should be independent (1.0)."""
        legs = [
            _leg(event_id="e1", sport="soccer_epl"),
            _leg(event_id="e2", sport="basketball_nba"),
        ]
        assert CorrelationEngine.compute_combo_correlation(legs) == 1.0

    def test_same_league_penalty(self):
        """Two EPL games should get league penalty."""
        legs = [
            _leg(event_id="e1", sport="soccer_epl"),
            _leg(event_id="e2", sport="soccer_epl"),
        ]
        mult = CorrelationEngine.compute_combo_correlation(legs)
        assert mult == pytest.approx(0.92)

    def test_same_sport_diff_league(self):
        """EPL + La Liga should get sport penalty."""
        legs = [
            _leg(event_id="e1", sport="soccer_epl"),
            _leg(event_id="e2", sport="soccer_laliga"),
        ]
        mult = CorrelationEngine.compute_combo_correlation(legs)
        assert mult == pytest.approx(0.97)


# ---------------------------------------------------------------------------
# Full combo: multiplier can exceed 1.0
# ---------------------------------------------------------------------------

class TestFullComboMultiplier:
    def test_positive_sgp_combo_boosts_probability(self):
        """A combo with positively correlated SGP legs should have multiplier > 1.0."""
        legs = [
            _leg(event_id="e1", selection="Home", odds=1.50, market="h2h"),
            _leg(event_id="e1", selection="Over 2.5", odds=1.90, market="totals 2.5"),
        ]
        mult = CorrelationEngine.compute_combo_correlation(legs)
        assert mult > 1.0, "Positive SGP should boost joint probability"

    def test_negative_sgp_combo_penalizes(self):
        """A combo with negatively correlated SGP legs should have multiplier < 1.0."""
        legs = [
            _leg(event_id="e1", selection="Home", odds=1.50, market="h2h"),
            _leg(event_id="e1", selection="Under 0.5", odds=4.0, market="totals 0.5"),
        ]
        mult = CorrelationEngine.compute_combo_correlation(legs)
        assert mult < 1.0, "Negative SGP should penalize joint probability"

    def test_multiplier_clamped(self):
        """Multiplier should be clamped to [0.50, 2.50]."""
        # Many positively correlated legs
        legs = [
            _leg(event_id=f"e{i}", selection="Home", odds=1.50, market="h2h")
            for i in range(20)
        ]
        mult = CorrelationEngine.compute_combo_correlation(legs)
        assert 0.50 <= mult <= 2.50
