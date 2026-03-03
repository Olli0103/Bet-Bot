"""Tests for src/core/betting_math.py — foundational EV, Kelly, and odds math."""
import pytest

from src.core.betting_math import (
    combo_odds,
    combo_probability,
    effective_tax_rate,
    expected_value,
    implied_probability,
    kelly_fraction,
    kelly_stake,
    public_bias_score,
)


# ---------------------------------------------------------------------------
# implied_probability
# ---------------------------------------------------------------------------

class TestImpliedProbability:
    def test_even_odds(self):
        assert implied_probability(2.0) == pytest.approx(0.5)

    def test_heavy_favorite(self):
        assert implied_probability(1.25) == pytest.approx(0.8)

    def test_long_shot(self):
        assert implied_probability(10.0) == pytest.approx(0.1)

    def test_odds_of_one(self):
        # Degenerate edge: 1/1.0 = 1.0
        assert implied_probability(1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# effective_tax_rate
# ---------------------------------------------------------------------------

class TestEffectiveTaxRate:
    def test_standard_tax(self):
        assert effective_tax_rate(0.05) == 0.05

    def test_tax_free_mode(self):
        assert effective_tax_rate(0.05, tax_free_mode=True) == 0.0

    def test_combo_3_legs_tax_free(self):
        assert effective_tax_rate(0.05, is_combo=True, combo_legs=3) == 0.0

    def test_combo_2_legs_still_taxed(self):
        assert effective_tax_rate(0.05, is_combo=True, combo_legs=2) == 0.05

    def test_combo_0_legs_taxed(self):
        assert effective_tax_rate(0.05, is_combo=True, combo_legs=0) == 0.05


# ---------------------------------------------------------------------------
# expected_value
# ---------------------------------------------------------------------------

class TestExpectedValue:
    def test_positive_ev(self):
        # 60% chance of winning at 2.0 odds → EV > 0
        ev = expected_value(0.6, 2.0, tax_rate=0.0)
        assert ev == pytest.approx(0.2)

    def test_negative_ev(self):
        # 40% chance of winning at 2.0 odds → EV < 0
        ev = expected_value(0.4, 2.0, tax_rate=0.0)
        assert ev == pytest.approx(-0.2)

    def test_break_even(self):
        # 50% chance at 2.0 odds → EV = 0
        ev = expected_value(0.5, 2.0, tax_rate=0.0)
        assert ev == pytest.approx(0.0)

    def test_with_tax(self):
        # 60% chance, 2.0 odds, 5% tax
        # net_profit = (2.0 - 1.0) * 0.95 = 0.95
        # EV = 0.6 * 0.95 - 0.4 = 0.57 - 0.4 = 0.17
        ev = expected_value(0.6, 2.0, tax_rate=0.05)
        assert ev == pytest.approx(0.17)

    def test_high_odds_positive(self):
        # 15% chance at 10.0 odds → EV = 0.15*9 - 0.85 = 0.5
        ev = expected_value(0.15, 10.0, tax_rate=0.0)
        assert ev == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------

class TestKellyFraction:
    def test_positive_edge(self):
        # 60% prob at 2.0 odds, frac=0.2
        # raw = (1.0 * 0.6 - 0.4) / 1.0 = 0.2
        # kelly = 0.2 * 0.2 = 0.04
        kf = kelly_fraction(0.6, 2.0, frac=0.2, tax_rate=0.0)
        assert kf == pytest.approx(0.04)

    def test_no_edge(self):
        # 50% prob at 2.0 odds → raw = 0.0
        kf = kelly_fraction(0.5, 2.0, frac=0.2, tax_rate=0.0)
        assert kf == pytest.approx(0.0)

    def test_negative_edge_returns_zero(self):
        # 30% prob at 2.0 odds → raw < 0 → clamped to 0
        kf = kelly_fraction(0.3, 2.0, frac=0.2, tax_rate=0.0)
        assert kf == 0.0

    def test_with_tax(self):
        # 60% prob, 2.0 odds, 5% tax
        # net_b = 1.0 * 0.95 = 0.95
        # raw = (0.95 * 0.6 - 0.4) / 0.95 = (0.57 - 0.4) / 0.95 = 0.1789
        # kelly = 0.1789 * 0.2 = 0.0358
        kf = kelly_fraction(0.6, 2.0, frac=0.2, tax_rate=0.05)
        assert kf == pytest.approx(0.0358, abs=0.001)

    def test_net_b_zero(self):
        # If odds <= 1.0 after tax → net_b = 0 → return 0
        kf = kelly_fraction(0.9, 1.0, frac=0.2, tax_rate=0.0)
        assert kf == 0.0


# ---------------------------------------------------------------------------
# kelly_stake
# ---------------------------------------------------------------------------

class TestKellyStake:
    def test_basic(self):
        assert kelly_stake(1000.0, 0.04) == pytest.approx(40.0)

    def test_zero_fraction(self):
        assert kelly_stake(1000.0, 0.0) == 0.0

    def test_negative_fraction_clamped(self):
        assert kelly_stake(1000.0, -0.05) == 0.0


# ---------------------------------------------------------------------------
# combo_odds / combo_probability
# ---------------------------------------------------------------------------

class TestComboMath:
    def test_combo_odds(self):
        assert combo_odds([2.0, 3.0, 1.5]) == pytest.approx(9.0)

    def test_combo_probability(self):
        assert combo_probability([0.5, 0.4, 0.8]) == pytest.approx(0.16)

    def test_combo_single_leg(self):
        assert combo_odds([2.5]) == pytest.approx(2.5)
        assert combo_probability([0.6]) == pytest.approx(0.6)

    def test_combo_empty(self):
        assert combo_odds([]) == pytest.approx(1.0)
        assert combo_probability([]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# public_bias_score
# ---------------------------------------------------------------------------

class TestPublicBiasScore:
    def test_shaded_favorite(self):
        # Tipico has lower odds (higher implied prob) than Pinnacle for selection
        sharp = {"Home": 2.0, "Away": 3.0}
        retail = {"Home": 1.8, "Away": 3.5}
        bias = public_bias_score(sharp, retail)
        # Home: retail_ip=1/1.8=0.5556, sharp_ip=1/2.0=0.5 → bias=+0.0556
        assert bias["Home"] > 0.02

    def test_generous_underdog(self):
        # Retail has higher odds (lower implied prob) for underdog
        sharp = {"Home": 2.0, "Away": 3.0}
        retail = {"Home": 1.8, "Away": 3.5}
        bias = public_bias_score(sharp, retail)
        # Away: retail_ip=1/3.5=0.2857, sharp_ip=1/3.0=0.3333 → bias=-0.0476
        assert bias["Away"] < -0.02

    def test_identical_markets(self):
        sharp = {"Home": 2.0, "Away": 2.0}
        retail = {"Home": 2.0, "Away": 2.0}
        bias = public_bias_score(sharp, retail)
        assert bias["Home"] == pytest.approx(0.0)

    def test_empty_markets(self):
        assert public_bias_score({}, {"Home": 2.0}) == {}
        assert public_bias_score({"Home": 2.0}, {}) == {}
