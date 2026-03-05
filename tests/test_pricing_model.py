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
    """Test the stabilized inverse-variance blending with metric scaling.

    CLV MSE is scaled by 4.0 to match Brier magnitude, and CLV weight
    is clamped to [0.10, 0.40] to prevent extreme allocations.
    """

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
    def test_metric_scaling_prevents_clv_dominance(self, mock_load):
        """CLV MSE 0.02 (raw) should NOT dominate brier 0.20 after 4x scaling.

        Without scaling: w_clv = (1/0.02) / (1/0.02 + 1/0.20) = 50/55 = 0.91
        With 4x scaling: clv_scaled = 0.08, w_clv = (1/0.08) / (1/0.08 + 1/0.20)
                        = 12.5/17.5 = 0.71 ... but clamped to max 0.40.
        """
        features = ["sharp_implied_prob"]
        classifier_data = self._make_mock_model(0.70, 0.20, features)
        clv_data = self._make_mock_clv(0.40, 0.02, features)  # Very low raw MSE

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
        # CLV weight clamped to 0.40, so classifier gets 0.60
        # Expected: 0.60 * 0.70 + 0.40 * 0.40 = 0.58
        assert result > 0.55, f"Classifier should still dominate, got {result}"

    @patch("src.core.pricing_model._load_joblib_model")
    def test_clv_weight_clamped_to_max_040(self, mock_load):
        """Even with very good CLV metrics, its weight should not exceed 0.40."""
        features = ["sharp_implied_prob"]
        classifier_data = self._make_mock_model(0.60, 0.25, features)  # poor brier
        clv_data = self._make_mock_clv(0.80, 0.01, features)  # excellent CLV

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
        # w_clv clamped to 0.40: 0.60 * 0.60 + 0.40 * 0.80 = 0.68
        expected = 0.60 * 0.60 + 0.40 * 0.80
        assert abs(result - expected) < 0.10

    @patch("src.core.pricing_model._load_joblib_model")
    def test_clv_weight_floored_at_010(self, mock_load):
        """Even with terrible CLV metrics, its weight should be at least 0.10."""
        features = ["sharp_implied_prob"]
        classifier_data = self._make_mock_model(0.70, 0.10, features)  # great brier
        clv_data = self._make_mock_clv(0.50, 0.50, features)  # terrible CLV MSE

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
        # w_clv floored at 0.10: 0.90 * 0.70 + 0.10 * 0.50 = 0.68
        expected = 0.90 * 0.70 + 0.10 * 0.50
        assert abs(result - expected) < 0.10

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

    @patch("src.core.pricing_model._load_joblib_model")
    def test_zero_metrics_dont_cause_division_error(self, mock_load):
        """Epsilon guard should prevent division by zero."""
        features = ["sharp_implied_prob"]
        classifier_data = self._make_mock_model(0.60, 0.0, features)  # zero brier!
        clv_data = self._make_mock_clv(0.50, 0.0, features)  # zero MSE!

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
        # Should not raise, should return a valid result
        result = model._xgboost_predict(
            {"sharp_implied_prob": 0.5}, "soccer_epl"
        )
        assert result is not None
        assert 0.01 <= result <= 0.99
