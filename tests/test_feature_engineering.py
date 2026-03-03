"""Tests for src/core/feature_engineering.py — feature construction."""
import pytest

from src.core.feature_engineering import FeatureEngineer


class TestCLVProxy:
    def test_positive_edge(self):
        # Target=2.5, Sharp=2.0 → (2.5/2.0)-1 = 0.25
        assert FeatureEngineer.calculate_clv_proxy(2.5, 2.0) == pytest.approx(0.25)

    def test_negative_edge(self):
        # Target=1.8, Sharp=2.0 → (1.8/2.0)-1 = -0.10
        assert FeatureEngineer.calculate_clv_proxy(1.8, 2.0) == pytest.approx(-0.10)

    def test_no_edge(self):
        assert FeatureEngineer.calculate_clv_proxy(2.0, 2.0) == pytest.approx(0.0)

    def test_invalid_sharp_odds(self):
        assert FeatureEngineer.calculate_clv_proxy(2.0, 1.0) == 0.0
        assert FeatureEngineer.calculate_clv_proxy(2.0, 0.5) == 0.0

    def test_invalid_target_odds(self):
        assert FeatureEngineer.calculate_clv_proxy(1.0, 2.0) == 0.0


class TestCalculateVig:
    def test_three_way_market(self):
        # Fair 3-way: sum of implied probs > 1
        market = {"Home": 2.0, "Draw": 3.5, "Away": 4.0}
        vig = FeatureEngineer.calculate_vig(market)
        # 0.5 + 0.2857 + 0.25 - 1.0 = 0.0357
        assert vig == pytest.approx(0.0357, abs=0.001)

    def test_two_way_market(self):
        market = {"Home": 1.9, "Away": 1.9}
        vig = FeatureEngineer.calculate_vig(market)
        # 0.5263 + 0.5263 - 1.0 = 0.0526
        assert vig == pytest.approx(0.0526, abs=0.001)

    def test_fair_market(self):
        # Perfectly fair: 1/2.0 + 1/2.0 = 1.0 → vig = 0
        assert FeatureEngineer.calculate_vig({"A": 2.0, "B": 2.0}) == pytest.approx(0.0)

    def test_empty_market(self):
        assert FeatureEngineer.calculate_vig({}) == 0.0

    def test_invalid_prices_skipped(self):
        assert FeatureEngineer.calculate_vig({"A": 0.5, "B": -1.0}) == 0.0


class TestBuildCoreFeatures:
    def test_home_selection(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"Bayern": 1.9, "Draw": 3.5, "Dortmund": 4.2},
            sentiment_home=0.5,
            sentiment_away=-0.3,
            injuries_home=1,
            injuries_away=3,
            selection="Bayern",
            home_team="Bayern",
        )
        assert f["sharp_implied_prob"] == pytest.approx(1 / 1.9, abs=0.001)
        assert f["home_advantage"] == 1.0
        # Sentiment delta: home - away = 0.5 - (-0.3) = 0.8
        assert f["sentiment_delta"] == pytest.approx(0.8)
        # Injury delta: away - home = 3 - 1 = 2 (positive = good for selected)
        assert f["injury_delta"] == pytest.approx(2.0)

    def test_away_selection(self):
        f = FeatureEngineer.build_core_features(
            target_odds=4.0,
            sharp_odds=4.2,
            sharp_market={"Bayern": 1.9, "Draw": 3.5, "Dortmund": 4.2},
            sentiment_home=0.5,
            sentiment_away=-0.3,
            injuries_home=1,
            injuries_away=3,
            selection="Dortmund",
            home_team="Bayern",
        )
        assert f["home_advantage"] == 0.0
        # Sentiment delta: away - home = -0.3 - 0.5 = -0.8
        assert f["sentiment_delta"] == pytest.approx(-0.8)
        # Injury delta: home - away = 1 - 3 = -2 (negative = bad for selected)
        assert f["injury_delta"] == pytest.approx(-2.0)

    def test_poisson_included_when_provided(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=2.0,
            sharp_market={"A": 2.0}, sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0, selection="A", home_team="A",
            poisson_true_prob=0.55,
        )
        assert f["poisson_true_prob"] == pytest.approx(0.55)

    def test_poisson_excluded_when_none(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=2.0,
            sharp_market={"A": 2.0}, sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0, selection="A", home_team="A",
        )
        assert "poisson_true_prob" not in f

    def test_clv_in_features(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.5, sharp_odds=2.0,
            sharp_market={"A": 2.0}, sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0, selection="A", home_team="A",
        )
        assert f["clv"] == pytest.approx(0.25)


class TestBuildTotalsFeatures:
    def test_basic_totals(self):
        f = FeatureEngineer.build_totals_features(
            over_odds=1.9, under_odds=1.9,
            sharp_over_odds=1.85, sharp_under_odds=1.95,
            point=2.5,
        )
        assert f["totals_point"] == 2.5
        assert f["sharp_over_prob"] == pytest.approx(1 / 1.85, abs=0.001)
        assert "poisson_over_prob" not in f

    def test_with_poisson(self):
        f = FeatureEngineer.build_totals_features(
            over_odds=1.9, under_odds=1.9,
            sharp_over_odds=1.85, sharp_under_odds=1.95,
            point=2.5, poisson_over_prob=0.60,
        )
        assert f["poisson_over_prob"] == pytest.approx(0.60)


class TestBuildSpreadsFeatures:
    def test_basic_spreads(self):
        f = FeatureEngineer.build_spreads_features(
            spread_odds=1.9, sharp_spread_odds=1.85, point=-3.5,
        )
        assert f["spread_point"] == -3.5
        assert f["sharp_spread_prob"] == pytest.approx(1 / 1.85, abs=0.001)
