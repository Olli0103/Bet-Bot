"""Tests for src/core/betting_math.py — foundational EV, Kelly, and odds math."""
import pytest

from src.core.betting_math import (
    DEFAULT_GERMAN_TAX_RATE,
    _remove_vig,
    combo_odds,
    combo_probability,
    effective_tax_rate,
    expected_value,
    get_net_payout,
    implied_probability,
    kelly_fraction,
    kelly_fraction_uncapped,
    kelly_stake,
    public_bias_score,
    tax_adjusted_expected_value,
    tax_adjusted_kelly_growth,
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
        # 60% chance, 2.0 odds, 5% tax on gross payout (German Tipico model)
        # net_odds = 2.0 * 0.95 = 1.90
        # net_profit = 1.90 - 1.0 = 0.90
        # EV = 0.6 * 0.90 - 0.4 = 0.54 - 0.4 = 0.14
        ev = expected_value(0.6, 2.0, tax_rate=0.05)
        assert ev == pytest.approx(0.14)

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
        # 60% prob, 2.0 odds, 5% tax on gross (German model)
        # net_b = 2.0 * 0.95 - 1.0 = 0.90
        # raw = (0.90 * 0.6 - 0.4) / 0.90 = 0.14/0.90 = 0.1556
        # kelly = 0.1556 * 0.2 = 0.0311
        kf = kelly_fraction(0.6, 2.0, frac=0.2, tax_rate=0.05)
        assert kf == pytest.approx(0.0311, abs=0.001)

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


# ---------------------------------------------------------------------------
# Kelly hard-cap safety
# ---------------------------------------------------------------------------

class TestKellyHardCap:
    def test_cap_applied_at_math_layer(self):
        """Kelly fraction is hard-capped at max_fraction (defense-in-depth)."""
        # 95% prob at 2.0 odds → raw Kelly = 0.9, * frac=1.0 = 0.9
        # Must be capped at 0.05 (default max_fraction)
        kf = kelly_fraction(0.95, 2.0, frac=1.0, tax_rate=0.0)
        assert kf <= 0.05

    def test_uncapped_returns_true_value(self):
        """kelly_fraction_uncapped returns the true value for ranking."""
        kf_uncapped = kelly_fraction_uncapped(0.95, 2.0, frac=1.0, tax_rate=0.0)
        kf_capped = kelly_fraction(0.95, 2.0, frac=1.0, tax_rate=0.0)
        assert kf_uncapped > kf_capped
        assert kf_uncapped > 0.5  # Should be around 0.9

    def test_custom_max_fraction(self):
        kf = kelly_fraction(0.95, 2.0, frac=1.0, tax_rate=0.0, max_fraction=0.10)
        assert kf <= 0.10

    def test_small_edge_not_affected(self):
        """Cap should not affect normal small-edge bets."""
        kf = kelly_fraction(0.55, 2.0, frac=0.2, tax_rate=0.0)
        # raw = 0.1, * 0.2 = 0.02 → well under 0.05 cap
        assert kf == pytest.approx(0.02)
        assert kf < 0.05


# ---------------------------------------------------------------------------
# Vig removal
# ---------------------------------------------------------------------------

class TestVigRemoval:
    def test_removes_vig_correctly(self):
        # Pinnacle h2h: 1.91, 2.00 → IPs: 0.5236, 0.5000 → total=1.0236
        prices = {"Home": 1.91, "Away": 2.00}
        fair = _remove_vig(prices)
        assert abs(fair["Home"] + fair["Away"] - 1.0) < 0.001
        assert fair["Home"] > fair["Away"]

    def test_empty_prices(self):
        assert _remove_vig({}) == {}

    def test_bias_score_uses_fair_probs(self):
        """Bias should be near-zero when both books have same fair odds
        but different margins."""
        # Same fair odds, different vig
        sharp = {"Home": 1.91, "Away": 1.91}  # ~4.7% vig
        retail = {"Home": 1.80, "Away": 1.80}  # ~11% vig
        bias = public_bias_score(sharp, retail)
        # Fair probs should be identical (0.5 each) → bias ~0
        assert abs(bias["Home"]) < 0.01


# ---------------------------------------------------------------------------
# German tax-aware functions (Tippprovider SOTA)
# ---------------------------------------------------------------------------

class TestDefaultGermanTaxRate:
    def test_constant_is_five_point_three_percent(self):
        assert DEFAULT_GERMAN_TAX_RATE == 0.053


class TestGetNetPayout:
    def test_standard_tax(self):
        # odds 2.0, tax 5.3% → 2.0 * 0.947 = 1.894
        assert get_net_payout(2.0) == pytest.approx(1.894)

    def test_no_tax(self):
        assert get_net_payout(2.0, tax_rate=0.0) == pytest.approx(2.0)

    def test_high_odds(self):
        # odds 10.0, tax 5.3% → 10.0 * 0.947 = 9.47
        assert get_net_payout(10.0) == pytest.approx(9.47)


class TestTaxAdjustedExpectedValue:
    def test_defaults_to_german_tax(self):
        """tax_adjusted_expected_value defaults to 5.3% tax unlike expected_value."""
        ev_no_tax = expected_value(0.6, 2.0, tax_rate=0.0)
        ev_tax = tax_adjusted_expected_value(0.6, 2.0)
        assert ev_tax < ev_no_tax
        # net_odds = 2.0 * 0.947 = 1.894, net_profit = 0.894
        # EV = 0.6 * 0.894 - 0.4 = 0.1364
        assert ev_tax == pytest.approx(0.1364)

    def test_matches_expected_value_when_same_rate(self):
        """Should match expected_value when given the same tax rate."""
        ev1 = expected_value(0.55, 3.0, tax_rate=0.053)
        ev2 = tax_adjusted_expected_value(0.55, 3.0, tax_rate=0.053)
        assert ev1 == pytest.approx(ev2)

    def test_losing_bet_after_tax(self):
        """A marginally +EV pre-tax bet becomes -EV after 5.3% Wettsteuer."""
        # 51% at 2.0: pre-tax EV = 0.51*1.0 - 0.49 = 0.02 (barely positive)
        # post-tax: 0.51*0.894 - 0.49 = -0.03406 (negative!)
        ev = tax_adjusted_expected_value(0.51, 2.0)
        assert ev < 0


class TestTaxAdjustedKellyGrowth:
    def test_positive_edge_positive_growth(self):
        """Strong edge should yield positive log-growth."""
        g = tax_adjusted_kelly_growth(0.6, 2.0, fraction=0.05)
        assert g > 0

    def test_no_edge_zero_growth(self):
        """Zero fraction should yield zero growth."""
        g = tax_adjusted_kelly_growth(0.6, 2.0, fraction=0.0)
        assert g == 0.0

    def test_negative_edge_negative_growth(self):
        """Bet with tax-negative EV should have negative growth at any fraction."""
        # 40% prob at 2.0 odds, 5.3% tax → definitely -EV
        g = tax_adjusted_kelly_growth(0.4, 2.0, fraction=0.05)
        assert g < 0

    def test_fraction_at_one_returns_zero(self):
        """Fraction exactly 1.0 should return 0 (boundary guard)."""
        g = tax_adjusted_kelly_growth(0.6, 2.0, fraction=1.0)
        assert g == 0.0

    def test_fraction_near_one_is_very_negative(self):
        """Fraction near 1.0 yields deeply negative growth (ruin risk)."""
        g = tax_adjusted_kelly_growth(0.6, 2.0, fraction=0.99)
        assert g < -1.0  # log(0.01) dominates on loss

    def test_net_odds_below_one(self):
        """If odds * (1-tax) <= 1.0, growth should be -inf."""
        import math
        g = tax_adjusted_kelly_growth(0.9, 1.0, fraction=0.05)
        assert g == -math.inf
