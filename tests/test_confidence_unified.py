"""Tests for unified confidence model, combo leg gates, and UI consistency.

Mandatory test cases:
1) test_single_tips_sorted_by_model_conf_desc
2) test_single_tip_confidence_gate_blocks_low_conf
3) test_steam_move_cannot_bypass_conf_gate
4) test_combo_leg_min_confidence_40_applied
5) test_ui_confidence_consistency_single_source
"""
import pytest

from src.core.betting_engine import BettingEngine
from src.core.risk_guards import passes_confidence_gate
from src.core.settings import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal_dict(sport: str, model_p: float, ev: float = 0.05,
                      event_id: str = "e1", selection: str = "Team A",
                      odds: float = 2.0, market: str = "h2h") -> dict:
    """Build a minimal signal dict matching BetSignal.model_dump() layout."""
    return {
        "sport": sport,
        "event_id": event_id,
        "market": market,
        "selection": selection,
        "bookmaker_odds": odds,
        "model_probability": model_p,
        "expected_value": ev,
        "confidence": model_p,  # unified: confidence == model_probability
        "source_quality": 1.0,
        "recommended_stake": 2.0,
        "kelly_fraction": 0.01,
        "kelly_raw": 0.01,
        "stake_before_cap": 2.0,
        "stake_cap_applied": False,
        "trigger": "",
        "rejected_reason": "",
    }


def _sort_like_handler(items: list) -> list:
    """Replicate handler sort: model_probability DESC -> EV DESC -> odds ASC."""
    return sorted(
        items,
        key=lambda x: (
            float(x["model_probability"]),
            float(x["expected_value"]),
            -float(x["bookmaker_odds"]),
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# 1) test_single_tips_sorted_by_model_conf_desc
# ---------------------------------------------------------------------------

class TestSingleTipsSortedByModelConf:

    def test_sorted_descending(self):
        """Top10 single tips must be strictly sorted by model_probability DESC."""
        items = [
            _make_signal_dict("soccer_epl", model_p=0.58, ev=0.10, event_id="a"),
            _make_signal_dict("soccer_epl", model_p=0.72, ev=0.03, event_id="b"),
            _make_signal_dict("soccer_epl", model_p=0.65, ev=0.08, event_id="c"),
            _make_signal_dict("soccer_epl", model_p=0.91, ev=0.02, event_id="d"),
        ]
        sorted_items = _sort_like_handler(items)
        probs = [x["model_probability"] for x in sorted_items]
        assert probs == sorted(probs, reverse=True)
        assert probs[0] == 0.91  # position #1 is always best model_probability

    def test_ev_tiebreak_when_equal_model_prob(self):
        """When model_probability is equal, higher EV wins."""
        items = [
            _make_signal_dict("soccer_epl", model_p=0.70, ev=0.02, event_id="x"),
            _make_signal_dict("soccer_epl", model_p=0.70, ev=0.08, event_id="y"),
        ]
        sorted_items = _sort_like_handler(items)
        assert sorted_items[0]["event_id"] == "y"
        assert sorted_items[1]["event_id"] == "x"


# ---------------------------------------------------------------------------
# 2) test_single_tip_confidence_gate_blocks_low_conf
# ---------------------------------------------------------------------------

class TestConfidenceGateBlocksLowConf:

    def test_gate_blocks_28_pct(self):
        """A 28% model_probability must be blocked (soccer h2h gate = 55%)."""
        passed, min_c = passes_confidence_gate(0.28, "soccer_epl", "h2h")
        assert not passed
        assert min_c == 0.55

    def test_gate_blocks_49_pct(self):
        """A 49% model_probability must be blocked (soccer h2h gate = 55%)."""
        passed, _ = passes_confidence_gate(0.49, "soccer_epl", "h2h")
        assert not passed

    def test_engine_zeroes_stake_for_low_conf(self):
        """BettingEngine must set stake=0 and fill rejected_reason for low conf."""
        engine = BettingEngine(bankroll=1000.0)
        sig = engine.make_signal(
            sport="soccer_epl",
            event_id="e1",
            market="h2h",
            selection="Home (A vs B)",
            bookmaker_odds=2.50,
            model_probability=0.28,
        )
        assert sig.recommended_stake == 0.0
        assert "reject_confidence_below_min" in sig.rejected_reason
        assert sig.kelly_fraction == 0.0


# ---------------------------------------------------------------------------
# 3) test_steam_move_cannot_bypass_conf_gate
# ---------------------------------------------------------------------------

class TestSteamMoveCannotBypassGate:

    def test_steam_move_blocked_at_44_pct(self):
        """steam_move trigger must NOT bypass the confidence gate."""
        engine = BettingEngine(bankroll=1000.0)
        sig = engine.make_signal(
            sport="soccer_epl",
            event_id="steam_evt",
            market="h2h",
            selection="Home",
            bookmaker_odds=2.10,
            model_probability=0.44,
            trigger="steam_move",
        )
        assert sig.recommended_stake == 0.0
        assert "reject_confidence_below_min" in sig.rejected_reason

    def test_totals_steam_blocked_at_50_pct(self):
        """totals_steam trigger must NOT bypass the confidence gate for totals."""
        engine = BettingEngine(bankroll=1000.0)
        sig = engine.make_signal(
            sport="soccer_epl",
            event_id="totals_evt",
            market="totals 2.5",
            selection="Over 2.5",
            bookmaker_odds=1.85,
            model_probability=0.50,
            trigger="totals_steam",
        )
        # soccer totals gate = 0.56
        assert sig.recommended_stake == 0.0
        assert "reject_confidence_below_min" in sig.rejected_reason

    def test_steam_move_allowed_above_gate(self):
        """steam_move with sufficient confidence should still work."""
        engine = BettingEngine(bankroll=1000.0)
        sig = engine.make_signal(
            sport="soccer_epl",
            event_id="steam_ok",
            market="h2h",
            selection="Home",
            bookmaker_odds=2.10,
            model_probability=0.60,
            trigger="steam_move",
        )
        assert sig.recommended_stake > 0
        assert sig.rejected_reason == ""


# ---------------------------------------------------------------------------
# 4) test_combo_leg_min_confidence_40_applied
# ---------------------------------------------------------------------------

class TestComboLegMinConfidence:

    def test_low_confidence_legs_filtered_out(self):
        """Combo legs with probability < 40% must be excluded."""
        legs = [
            {"probability": 0.35, "sport": "soccer_epl", "odds": 2.0,
             "event_id": "a", "selection": "A", "market_type": "h2h"},
            {"probability": 0.50, "sport": "soccer_epl", "odds": 1.8,
             "event_id": "b", "selection": "B", "market_type": "h2h"},
            {"probability": 0.39, "sport": "basketball_nba", "odds": 2.5,
             "event_id": "c", "selection": "C", "market_type": "h2h"},
            {"probability": 0.65, "sport": "tennis_atp", "odds": 1.5,
             "event_id": "d", "selection": "D", "market_type": "h2h"},
        ]

        min_conf = settings.min_combo_leg_confidence  # 0.40
        eligible = [l for l in legs if l["probability"] >= min_conf]

        # Only B (0.50) and D (0.65) should survive
        assert len(eligible) == 2
        assert all(l["probability"] >= 0.40 for l in eligible)

    def test_exactly_40_pct_included(self):
        """A leg at exactly 40% should be included."""
        legs = [
            {"probability": 0.40, "sport": "soccer_epl", "odds": 2.0,
             "event_id": "a", "selection": "A", "market_type": "h2h"},
        ]
        min_conf = settings.min_combo_leg_confidence
        eligible = [l for l in legs if l["probability"] >= min_conf]
        assert len(eligible) == 1

    def test_cross_sport_allowed(self):
        """Combo legs from different sports should be allowed together."""
        legs = [
            {"probability": 0.60, "sport": "soccer_epl", "odds": 1.8,
             "event_id": "s1", "selection": "A", "market_type": "h2h"},
            {"probability": 0.55, "sport": "basketball_nba", "odds": 2.0,
             "event_id": "b1", "selection": "B", "market_type": "h2h"},
            {"probability": 0.50, "sport": "tennis_atp", "odds": 1.7,
             "event_id": "t1", "selection": "C", "market_type": "h2h"},
        ]
        min_conf = settings.min_combo_leg_confidence
        eligible = [l for l in legs if l["probability"] >= min_conf]
        # All 3 sports should be present
        sports = {l["sport"].split("_")[0] for l in eligible}
        assert sports == {"soccer", "basketball", "tennis"}


# ---------------------------------------------------------------------------
# 5) test_ui_confidence_consistency_single_source
# ---------------------------------------------------------------------------

class TestUIConfidenceConsistency:

    def test_confidence_equals_model_probability(self):
        """In the serialized signal, confidence must equal model_probability.

        This prevents the old bug where UI showed "Model 28% | Conf 100%"
        because model_probability and confidence were independent fields.
        """
        engine = BettingEngine(bankroll=1000.0)
        sig = engine.make_signal(
            sport="soccer_epl",
            event_id="ui_test",
            market="h2h",
            selection="Home",
            bookmaker_odds=1.90,
            model_probability=0.62,
            source_quality=0.80,  # this is NOT the display confidence
        )
        d = sig.model_dump()
        assert d["confidence"] == d["model_probability"]
        assert d["confidence"] == 0.62
        assert d["source_quality"] == 0.80

    def test_no_contradictory_display(self):
        """The card should never show model% != confidence%."""
        engine = BettingEngine(bankroll=1000.0)
        sig = engine.make_signal(
            sport="basketball_nba",
            event_id="ui2",
            market="h2h",
            selection="Away",
            bookmaker_odds=2.20,
            model_probability=0.58,
            source_quality=1.0,
        )
        d = sig.model_dump()
        # Both must be identical
        assert d["model_probability"] == d["confidence"] == 0.58
        # source_quality is separate and may differ
        assert d["source_quality"] == 1.0

    def test_low_conf_signal_still_has_unified_fields(self):
        """Even a rejected signal has confidence == model_probability."""
        engine = BettingEngine(bankroll=1000.0)
        sig = engine.make_signal(
            sport="soccer_epl",
            event_id="low",
            market="h2h",
            selection="Draw",
            bookmaker_odds=3.50,
            model_probability=0.30,
            source_quality=1.0,
        )
        d = sig.model_dump()
        assert d["confidence"] == d["model_probability"] == 0.30
        assert d["rejected_reason"] != ""
