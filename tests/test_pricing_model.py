"""Tests for src/core/pricing_model.py — probability predictions."""
from unittest.mock import patch, MagicMock

import numpy as np
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

    def test_icehockey(self):
        assert _sport_group("icehockey_nhl") == "icehockey"

    def test_americanfootball(self):
        assert _sport_group("americanfootball_nfl") == "americanfootball"

    def test_unknown_defaults_general(self):
        assert _sport_group("") == "general"
        assert _sport_group("cricket_ipl") == "general"


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


class TestInverseVarianceBlending:
    """Test the inverse-variance weighted blending of classifier and CLV."""

    def _make_mock_model(self, prob, brier, features):
        """Create a mock model artifact with predict_proba."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[1 - prob, prob]])
        return {
            "model": mock_model,
            "features": features,
            "metrics": {"brier_score": brier},
            "n_samples": 1000,
        }

    def _make_mock_clv(self, pred, mse, features):
        """Create a mock CLV regressor artifact."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([pred])
        return {
            "model": mock_model,
            "features": features,
            "metrics": {"clv_mse": mse, "clv_mse_baseline": mse + 0.01},
        }

    @patch("src.core.pricing_model._load_joblib_model")
    def test_equal_variance_gives_equal_weights(self, mock_load):
        """When brier_score == clv_mse, weights should be 50/50."""
        features = ["sharp_implied_prob"]
        classifier_data = self._make_mock_model(0.60, 0.20, features)
        clv_data = self._make_mock_clv(0.40, 0.20, features)

        def loader(group):
            if group == "soccer":
                return classifier_data
            if group == "clv_general":
                return clv_data
            return None

        mock_load.side_effect = loader

        # Clear cache
        from src.core import pricing_model
        pricing_model._model_cache.clear()

        model = QuantPricingModel(weights_file="__nonexistent__.json")
        result = model._xgboost_predict(
            {"sharp_implied_prob": 0.5}, "soccer_epl"
        )

        assert result is not None
        # With equal variance: 50/50 blend of 0.60 and 0.40 = 0.50
        assert abs(result - 0.50) < 0.05

    @patch("src.core.pricing_model._load_joblib_model")
    def test_better_classifier_gets_higher_weight(self, mock_load):
        """Classifier with much lower brier should dominate the blend."""
        features = ["sharp_implied_prob"]
        classifier_data = self._make_mock_model(0.70, 0.10, features)  # low brier
        clv_data = self._make_mock_clv(0.50, 0.30, features)  # high mse

        def loader(group):
            if group == "soccer":
                return classifier_data
            if group == "clv_general":
                return clv_data
            return None

        mock_load.side_effect = loader

        from src.core import pricing_model
        pricing_model._model_cache.clear()

        model = QuantPricingModel(weights_file="__nonexistent__.json")
        result = model._xgboost_predict(
            {"sharp_implied_prob": 0.5}, "soccer_epl"
        )

        assert result is not None
        # Classifier weight = (1/0.10) / (1/0.10 + 1/0.30) = 10/13.33 = 0.75
        # Expected: 0.75 * 0.70 + 0.25 * 0.50 = 0.65
        assert result > 0.60  # Classifier dominates

    @patch("src.core.pricing_model._load_joblib_model")
    def test_no_clv_returns_classifier_only(self, mock_load):
        """Without CLV model, should return classifier prob directly."""
        features = ["sharp_implied_prob"]
        classifier_data = self._make_mock_model(0.65, 0.20, features)

        def loader(group):
            if group == "soccer":
                return classifier_data
            return None

        mock_load.side_effect = loader

        from src.core import pricing_model
        pricing_model._model_cache.clear()

        model = QuantPricingModel(weights_file="__nonexistent__.json")
        result = model._xgboost_predict(
            {"sharp_implied_prob": 0.5}, "soccer_epl"
        )

        assert result is not None
        assert abs(result - 0.65) < 0.05
