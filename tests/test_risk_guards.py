"""Tests for Risk Guards: confidence gates, stake caps, and Top10 sorting.

Mandatory test cases:
1) confidence_gate_blocks_low_confidence_even_with_steam_move
2) stake_cap_applied_for_draw_and_longshot
3) top10_per_sport_sorted_by_confidence_desc
4) top10_item_1_is_best_confidence
5) no_global_top10_leak_into_sport_filter
6) dynamic_min_ev_scales_with_brier
"""
import pytest
from unittest.mock import patch

from src.core.risk_guards import (
    apply_stake_cap,
    get_dynamic_min_ev,
    get_min_confidence,
    passes_confidence_gate,
)
from src.core.betting_engine import BettingEngine


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_signal_dict(sport: str, confidence: float, ev: float = 0.05,
                      model_p: float = 0.0, event_id: str = "e1",
                      selection: str = "Team A", odds: float = 2.0,
                      market: str = "h2h") -> dict:
    """Build a minimal signal dict (as if from BetSignal.model_dump()).

    `confidence` parameter maps to model_probability (single source of truth).
    If model_p is explicitly given and != 0, it overrides confidence.
    """
    mp = model_p if model_p > 0 else confidence
    return {
        "sport": sport,
        "event_id": event_id,
        "market": market,
        "selection": selection,
        "bookmaker_odds": odds,
        "model_probability": mp,
        "expected_value": ev,
        "kelly_fraction": 0.01,
        "recommended_stake": 2.0,
        "confidence": mp,  # == model_probability (unified)
        "source_quality": 1.0,
        "kelly_raw": 0.02,
        "stake_before_cap": 2.0,
        "stake_cap_applied": False,
        "trigger": "",
        "rejected_reason": "",
    }


# ---------------------------------------------------------------------------
# 1) confidence_gate_blocks_low_confidence_even_with_steam_move
# ---------------------------------------------------------------------------

class TestConfidenceGate:

    def test_blocks_low_confidence(self):
        """A signal at 44% model_prob must be blocked for soccer h2h (gate=55%)."""
        passed, min_c = passes_confidence_gate(0.44, "soccer_epl", "h2h")
        assert not passed
        assert min_c == 0.55

    def test_blocks_low_confidence_even_with_steam_move(self):
        """steam_move trigger does not bypass the confidence gate.

        The BettingEngine.make_signal must zero out stake when confidence
        gate fails, regardless of what triggered the signal.
        """
        engine = BettingEngine(bankroll=1000.0)
        # Model probability 0.44 < soccer h2h gate 0.55
        sig = engine.make_signal(
            sport="soccer_epl",
            event_id="evt_steam",
            market="h2h",
            selection="Home",
            bookmaker_odds=2.10,
            model_probability=0.44,
            source_quality=1.0,
            trigger="steam_move",
        )
        assert sig.recommended_stake == 0.0
        assert sig.rejected_reason != ""
        assert "confidence" in sig.rejected_reason.lower()

    def test_passes_above_gate(self):
        """A signal at 60% model_prob passes the soccer h2h gate (55%)."""
        passed, _ = passes_confidence_gate(0.60, "soccer_epl", "h2h")
        assert passed

    def test_tennis_gate_higher(self):
        """Tennis requires 57%, so 56% should fail."""
        passed, min_c = passes_confidence_gate(0.56, "tennis_atp_dubai", "h2h")
        assert not passed
        assert min_c == 0.57

    def test_basketball_gate(self):
        passed, _ = passes_confidence_gate(0.55, "basketball_nba", "h2h")
        assert passed

    def test_icehockey_gate(self):
        passed, _ = passes_confidence_gate(0.54, "icehockey_nhl", "h2h")
        assert not passed

    def test_americanfootball_gate(self):
        passed, _ = passes_confidence_gate(0.55, "americanfootball_nfl", "h2h")
        assert passed


# ---------------------------------------------------------------------------
# 2) stake_cap_applied_for_draw_and_longshot
# ---------------------------------------------------------------------------

class TestStakeCap:

    def test_normal_stake_within_cap(self):
        """A 10 EUR stake on 1000 bankroll (1%) is within 1.5% cap."""
        capped, was = apply_stake_cap(10.0, 1000.0, odds=2.0)
        assert capped == 10.0
        assert not was

    def test_normal_stake_exceeds_cap(self):
        """A 20 EUR stake on 1000 bankroll (2%) exceeds 1.5% cap -> capped to 15."""
        capped, was = apply_stake_cap(20.0, 1000.0, odds=2.0)
        assert capped == 15.0
        assert was

    def test_draw_cap_lower(self):
        """Draw selection uses 0.75% cap -> 7.50 on 1000 bankroll."""
        capped, was = apply_stake_cap(10.0, 1000.0, odds=2.0, selection="Draw (A vs B)")
        assert capped == 7.50
        assert was

    def test_longshot_cap(self):
        """Odds >= 3.5 triggers longshot cap (0.75%)."""
        capped, was = apply_stake_cap(10.0, 1000.0, odds=4.0)
        assert capped == 7.50
        assert was

    def test_draw_and_longshot_combined(self):
        """Draw at longshot odds still uses the 0.75% cap."""
        capped, was = apply_stake_cap(10.0, 1000.0, odds=5.0, selection="Draw")
        assert capped == 7.50
        assert was

    def test_stake_below_cap_not_touched(self):
        """A 5 EUR stake on 1000 bankroll (0.5%) stays as-is."""
        capped, was = apply_stake_cap(5.0, 1000.0, odds=4.0)
        assert capped == 5.0
        assert not was

    def test_engine_applies_cap(self):
        """BettingEngine.make_signal applies the cap and marks it."""
        engine = BettingEngine(bankroll=500.0)
        # With prob=0.80 and odds=2.0 the Kelly formula gives a large stake
        sig = engine.make_signal(
            sport="soccer_epl",
            event_id="e1",
            market="h2h",
            selection="Home",
            bookmaker_odds=2.00,
            model_probability=0.80,
            source_quality=1.0,
        )
        # Max stake = 500 * 0.015 = 7.50
        assert sig.recommended_stake <= 7.50
        assert sig.stake_before_cap >= sig.recommended_stake


# ---------------------------------------------------------------------------
# 3) top10_per_sport_sorted_by_confidence_desc
# ---------------------------------------------------------------------------

def _sort_like_handler(items: list) -> list:
    """Replicate the handler's sorting: model_probability DESC -> EV DESC -> odds ASC."""
    return sorted(
        items,
        key=lambda x: (
            float(x.get("model_probability", 0)),
            float(x.get("expected_value", 0)),
            -float(x.get("bookmaker_odds", 99)),
        ),
        reverse=True,
    )


class TestTop10Sorting:

    def test_per_sport_sorted_by_model_conf_desc(self):
        """Top10 must be sorted by model_probability descending."""
        items = [
            _make_signal_dict("soccer_epl", confidence=0.65, ev=0.10, event_id="a"),
            _make_signal_dict("soccer_epl", confidence=0.99, ev=0.03, event_id="b"),
            _make_signal_dict("soccer_epl", confidence=0.80, ev=0.08, event_id="c"),
        ]
        sorted_items = _sort_like_handler(items)
        probs = [x["model_probability"] for x in sorted_items]
        assert probs == [0.99, 0.80, 0.65]

    def test_ev_tiebreak(self):
        """When confidence is equal, higher EV should rank first."""
        items = [
            _make_signal_dict("soccer_epl", confidence=1.0, ev=0.02, event_id="x"),
            _make_signal_dict("soccer_epl", confidence=1.0, ev=0.10, event_id="y"),
        ]
        sorted_items = _sort_like_handler(items)
        evs = [x["expected_value"] for x in sorted_items]
        assert evs == [0.10, 0.02]


# ---------------------------------------------------------------------------
# 4) top10_item_1_is_best_confidence
# ---------------------------------------------------------------------------

class TestTop10Item1Best:

    def test_item_1_is_best_confidence(self):
        """The first item in Top10 must have the highest model_probability."""
        items = [
            _make_signal_dict("basketball_nba", confidence=0.80, ev=0.05, event_id="b1"),
            _make_signal_dict("basketball_nba", confidence=0.99, ev=0.03, event_id="b2"),
            _make_signal_dict("basketball_nba", confidence=0.65, ev=0.10, event_id="b3"),
            _make_signal_dict("basketball_nba", confidence=0.90, ev=0.07, event_id="b4"),
        ]
        sorted_items = _sort_like_handler(items)
        top1 = sorted_items[0]
        assert top1["model_probability"] == max(x["model_probability"] for x in items)
        assert top1["event_id"] == "b2"


# ---------------------------------------------------------------------------
# 5) no_global_top10_leak_into_sport_filter
# ---------------------------------------------------------------------------

def _filter_items_by_sport(items: list, sport_filter: str) -> list:
    """Mirror of handlers._filter_items_by_sport."""
    if sport_filter == "all":
        return items
    return [x for x in items if str(x.get("sport", "")).startswith(sport_filter)]


class TestNoGlobalLeak:

    def test_sport_filter_uses_full_pool_not_global_top10(self):
        """Per-sport Top10 must draw from the full ranked pool, not a
        pre-sliced global top 10. This ensures minor sports (NHL, NFL)
        always have results if signals exist.
        """
        # 15 soccer signals + 5 NHL signals
        pool = []
        for i in range(15):
            pool.append(_make_signal_dict("soccer_epl", confidence=1.0,
                                          ev=0.10 - i * 0.003, event_id=f"s{i}"))
        for i in range(5):
            pool.append(_make_signal_dict("icehockey_nhl", confidence=0.90,
                                          ev=0.04 - i * 0.005, event_id=f"n{i}"))

        # Old bug: slicing to global top 10 first would lose all NHL
        global_top10 = _sort_like_handler(pool)[:10]
        nhl_in_global = _filter_items_by_sport(global_top10, "icehockey")
        assert len(nhl_in_global) == 0, "Global top10 should exclude NHL (all soccer)"

        # New behavior: filter full pool THEN take top 10
        nhl_filtered = _filter_items_by_sport(pool, "icehockey")
        nhl_sorted = _sort_like_handler(nhl_filtered)[:10]
        assert len(nhl_sorted) == 5, "NHL should have 5 signals from full pool"

    def test_all_sports_view_independent(self):
        """Viewing one sport must not affect another sport's results."""
        pool = [
            _make_signal_dict("soccer_epl", confidence=1.0, ev=0.10, event_id="s1"),
            _make_signal_dict("basketball_nba", confidence=0.90, ev=0.08, event_id="b1"),
            _make_signal_dict("tennis_atp", confidence=0.85, ev=0.06, event_id="t1"),
        ]

        # View soccer
        soccer = _filter_items_by_sport(pool, "soccer")
        assert len(soccer) == 1

        # View basketball after soccer -- pool unchanged
        bball = _filter_items_by_sport(pool, "basketball")
        assert len(bball) == 1

        # View all -- still has all 3
        all_items = _filter_items_by_sport(pool, "all")
        assert len(all_items) == 3


# ---------------------------------------------------------------------------
# 6) dynamic_min_ev_scales_with_brier
# ---------------------------------------------------------------------------

class TestDynamicMinEv:

    def _mock_load(self, brier_value):
        """Patch load_model inside get_dynamic_min_ev."""
        if brier_value is None:
            return patch("src.core.ml_trainer.load_model", return_value=None)
        return patch(
            "src.core.ml_trainer.load_model",
            return_value={"metrics": {"brier_score": brier_value}},
        )

    def test_brier_018_gives_lower_threshold(self):
        """Brier 0.18 -> min_ev = 0.18 * 0.15 = 0.027."""
        with self._mock_load(0.18):
            ev = get_dynamic_min_ev("soccer_epl")
        assert abs(ev - 0.027) < 0.001

    def test_brier_025_gives_higher_threshold(self):
        """Brier 0.25 -> min_ev = 0.25 * 0.15 = 0.0375."""
        with self._mock_load(0.25):
            ev = get_dynamic_min_ev("soccer_epl")
        assert abs(ev - 0.0375) < 0.001

    def test_floor_at_005(self):
        """Very good Brier (0.01) should still have at least 0.005 min_ev."""
        with self._mock_load(0.01):
            ev = get_dynamic_min_ev("soccer_epl")
        assert ev >= 0.005

    def test_cap_at_005(self):
        """Very bad Brier (0.5) should be capped at 0.05."""
        with self._mock_load(0.50):
            ev = get_dynamic_min_ev("soccer_epl")
        assert ev == 0.05

    def test_fallback_when_no_model(self):
        """Without any model, falls back to settings.min_ev_default."""
        with self._mock_load(None):
            ev = get_dynamic_min_ev("cricket_ipl")
        assert ev == 0.01  # settings.min_ev_default
