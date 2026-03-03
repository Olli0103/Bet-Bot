"""Tests for src/core/pricing_model.py — probability predictions."""
import pytest

from src.core.pricing_model import QuantPricingModel, _sport_group


class TestSportGroup:
    def test_soccer(self):
        assert _sport_group("soccer_epl") == "soccer"
        assert _sport_group("soccer_germany_bundesliga") == "soccer"

    def test_football_alias(self):
        assert _sport_group("football_epl") == "soccer"

    def test_basketball(self):
        assert _sport_group("basketball_nba") == "basketball"

    def test_tennis(self):
        assert _sport_group("tennis_atp") == "tennis"

    def test_unknown_defaults_general(self):
        assert _sport_group("icehockey_nhl") == "general"
        assert _sport_group("americanfootball_nfl") == "general"
        assert _sport_group("") == "general"


class TestLegacyPredict:
    """Test the legacy logistic regression fallback (no XGBoost models needed)."""

    def setup_method(self):
        self.model = QuantPricingModel(weights_file="__nonexistent__.json")

    def test_sharp_prob_passthrough(self):
        """With default weights (sharp_implied_prob=1.0, rest=0.0), legacy predict
        should approximate the input sharp_prob through the logistic function."""
        # sharp_prob=0.5 → log_odds=0.5 → sigmoid(0.5)=0.6225
        prob = self.model._legacy_predict(sharp_prob=0.5)
        assert 0.5 < prob < 0.7  # sigmoid(0.5)

    def test_extreme_high(self):
        # Very high sharp_prob → should still be capped by sigmoid
        prob = self.model._legacy_predict(sharp_prob=0.95)
        assert prob < 1.0
        assert prob > 0.5

    def test_extreme_low(self):
        prob = self.model._legacy_predict(sharp_prob=0.05)
        assert prob > 0.0
        assert prob < 0.6

    def test_zero_sharp_prob(self):
        prob = self.model._legacy_predict(sharp_prob=0.0)
        # sigmoid(0) = 0.5
        assert prob == pytest.approx(0.5)


class TestGetTrueProbability:
    """Test the main probability entry point (falls back to legacy without models)."""

    def setup_method(self):
        self.model = QuantPricingModel(weights_file="__nonexistent__.json")

    def test_without_features_uses_legacy(self):
        prob = self.model.get_true_probability(
            sharp_prob=0.5, sentiment=0.0, injuries=0.0,
        )
        assert 0.01 <= prob <= 0.99

    def test_with_empty_features_falls_back(self):
        """No .joblib models exist → falls back to legacy."""
        prob = self.model.get_true_probability(
            sharp_prob=0.5, features={"sharp_implied_prob": 0.5},
            sport="soccer_epl",
        )
        assert 0.01 <= prob <= 0.99

    def test_probability_bounded(self):
        """Probability must always be in [0.01, 0.99] (or legacy sigmoid range)."""
        for p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            prob = self.model.get_true_probability(sharp_prob=p)
            assert 0.0 < prob < 1.0
