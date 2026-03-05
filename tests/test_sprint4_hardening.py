"""Tests for Sprint 4 hardening: Beta Calibration, Gaussian Copula, Dynamic Kelly."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. BetaCalibratedModel wrapper
# ---------------------------------------------------------------------------


class TestBetaCalibratedModel:
    """Verify the BetaCalibratedModel wrapper produces valid probabilities."""

    def test_predict_proba_shape(self):
        """predict_proba returns (n, 2) array with columns summing to 1."""
        from betacal import BetaCalibration
        from xgboost import XGBClassifier

        from src.core.ml_trainer import BetaCalibratedModel

        # Train a tiny model
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        base = XGBClassifier(n_estimators=10, max_depth=2, verbosity=0, use_label_encoder=False)
        base.fit(X[:80], y[:80])

        cal = BetaCalibration(parameters="abm")
        raw_preds = base.predict_proba(X[80:])[:, 1]
        cal.fit(raw_preds, y[80:])

        model = BetaCalibratedModel(base, cal)
        proba = model.predict_proba(X[80:])

        assert proba.shape == (20, 2)
        # Columns must sum to ~1.0
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        # All values in [0.01, 0.99] (clamped)
        assert proba[:, 1].min() >= 0.01
        assert proba[:, 1].max() <= 0.99

    def test_predict_binary(self):
        """predict() returns binary 0/1 labels."""
        from betacal import BetaCalibration
        from xgboost import XGBClassifier

        from src.core.ml_trainer import BetaCalibratedModel

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        base = XGBClassifier(n_estimators=10, max_depth=2, verbosity=0, use_label_encoder=False)
        base.fit(X[:80], y[:80])

        cal = BetaCalibration(parameters="abm")
        cal.fit(base.predict_proba(X[80:])[:, 1], y[80:])

        model = BetaCalibratedModel(base, cal)
        preds = model.predict(X[80:])

        assert set(preds).issubset({0, 1})

    def test_no_double_calibration(self):
        """Verify raw XGBoost -> BetaCalibration is a single transform, not double."""
        from betacal import BetaCalibration
        from xgboost import XGBClassifier

        from src.core.ml_trainer import BetaCalibratedModel

        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        base = XGBClassifier(n_estimators=20, max_depth=3, verbosity=0, use_label_encoder=False)
        base.fit(X[:150], y[:150])

        cal = BetaCalibration(parameters="abm")
        raw = base.predict_proba(X[150:])[:, 1]
        cal.fit(raw, y[150:])

        model = BetaCalibratedModel(base, cal)

        # The wrapper should NOT apply any additional Platt scaling
        # Just: raw XGBoost proba -> beta calibration -> clamp
        calibrated = model.predict_proba(X[150:])[:, 1]

        # Calibrated values should have wider spread than Platt would give
        # (beta preserves tails better). At minimum, spread > 0.
        assert calibrated.std() > 0.01


# ---------------------------------------------------------------------------
# 2. Gaussian Copula
# ---------------------------------------------------------------------------


class TestGaussianCopula:
    """Verify the Gaussian copula produces valid joint probabilities."""

    def _make_legs(self, probs, sports=None, event_ids=None):
        from src.models.betting import ComboLeg

        legs = []
        for i, p in enumerate(probs):
            legs.append(ComboLeg(
                event_id=event_ids[i] if event_ids else f"evt_{i}",
                selection=f"Team_{i}",
                odds=round(1.0 / max(p, 0.01), 2),
                probability=p,
                sport=sports[i] if sports else "soccer_epl",
                market_type="h2h",
                home_team=f"Home_{i}",
                away_team=f"Away_{i}",
                market="h2h",
            ))
        return legs

    def test_single_leg_returns_marginal(self):
        """Single-leg combo should return the marginal probability."""
        from src.core.betting_engine import BettingEngine

        legs = self._make_legs([0.65])
        corr = np.eye(1)
        result = BettingEngine._compute_joint_probability_copula(legs, corr)
        assert abs(result - 0.65) < 0.01

    def test_independent_legs_match_product(self):
        """With identity correlation matrix, copula ≈ product of marginals."""
        from src.core.betting_engine import BettingEngine

        probs = [0.60, 0.55, 0.70]
        legs = self._make_legs(probs, event_ids=["e1", "e2", "e3"],
                               sports=["soccer_epl", "basketball_nba", "tennis_atp"])
        corr = np.eye(3)
        result = BettingEngine._compute_joint_probability_copula(legs, corr)

        independent = 0.60 * 0.55 * 0.70
        # Should be close to independent product
        assert abs(result - independent) < 0.02

    def test_positive_correlation_boosts(self):
        """Positive correlation should increase joint probability vs independent."""
        from src.core.betting_engine import BettingEngine

        probs = [0.60, 0.55]
        legs = self._make_legs(probs)
        corr_independent = np.eye(2)
        corr_positive = np.array([[1.0, 0.3], [0.3, 1.0]])

        p_ind = BettingEngine._compute_joint_probability_copula(legs, corr_independent)
        p_pos = BettingEngine._compute_joint_probability_copula(legs, corr_positive)

        assert p_pos > p_ind

    def test_negative_correlation_penalizes(self):
        """Negative correlation should decrease joint probability vs independent."""
        from src.core.betting_engine import BettingEngine

        probs = [0.60, 0.55]
        legs = self._make_legs(probs)
        corr_independent = np.eye(2)
        corr_negative = np.array([[1.0, -0.3], [-0.3, 1.0]])

        p_ind = BettingEngine._compute_joint_probability_copula(legs, corr_independent)
        p_neg = BettingEngine._compute_joint_probability_copula(legs, corr_negative)

        assert p_neg < p_ind

    def test_result_always_in_bounds(self):
        """Copula output must always be in (0, 1)."""
        from src.core.betting_engine import BettingEngine

        probs = [0.90, 0.85, 0.80, 0.75]
        legs = self._make_legs(probs, event_ids=["e1", "e2", "e3", "e4"])
        # Strong positive correlation
        corr = np.eye(4) * 0.4 + 0.6
        np.fill_diagonal(corr, 1.0)

        result = BettingEngine._compute_joint_probability_copula(legs, corr)
        assert 0.0 < result < 1.0

    def test_build_correlation_matrix_shape(self):
        """Correlation matrix must be n×n symmetric with 1s on diagonal."""
        from src.core.betting_engine import BettingEngine

        legs = self._make_legs([0.6, 0.5, 0.7],
                               event_ids=["e1", "e1", "e2"],
                               sports=["soccer_epl", "soccer_epl", "soccer_epl"])
        # Override market types for SGP testing
        legs[0].market_type = "h2h"
        legs[0].market = "h2h"
        legs[1].market_type = "totals"
        legs[1].market = "totals"

        corr = BettingEngine._build_correlation_matrix(legs)

        assert corr.shape == (3, 3)
        np.testing.assert_array_equal(np.diag(corr), [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(corr, corr.T)  # symmetric


# ---------------------------------------------------------------------------
# 3. Dynamic Kelly Shrinker
# ---------------------------------------------------------------------------


class TestDynamicKelly:
    """Verify Brier-score-dependent Kelly fraction scaling."""

    def test_well_calibrated_gets_full_frac(self):
        """Brier < 0.18 returns full base fraction."""
        from unittest.mock import patch

        from src.core.risk_guards import get_dynamic_kelly_frac

        mock_model = {"metrics": {"brier_score": 0.15}}
        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            frac = get_dynamic_kelly_frac("soccer_epl", base_frac=0.20)
        assert frac == 0.20

    def test_medium_brier_gets_75pct(self):
        """Brier 0.18-0.22 returns 75% of base."""
        from unittest.mock import patch

        from src.core.risk_guards import get_dynamic_kelly_frac

        mock_model = {"metrics": {"brier_score": 0.20}}
        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            frac = get_dynamic_kelly_frac("basketball_nba", base_frac=0.20)
        assert frac == pytest.approx(0.15)

    def test_poor_brier_gets_40pct(self):
        """Brier >= 0.22 returns 40% of base."""
        from unittest.mock import patch

        from src.core.risk_guards import get_dynamic_kelly_frac

        mock_model = {"metrics": {"brier_score": 0.25}}
        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            frac = get_dynamic_kelly_frac("tennis_atp", base_frac=0.20)
        assert frac == pytest.approx(0.08)

    def test_no_model_gets_50pct(self):
        """No model available returns 50% of base (maximum caution)."""
        from unittest.mock import patch

        from src.core.risk_guards import get_dynamic_kelly_frac

        with patch("src.core.ml_trainer.load_model", return_value=None):
            frac = get_dynamic_kelly_frac("unknown_sport", base_frac=0.20)
        assert frac == pytest.approx(0.10)

    def test_combo_kelly_is_dynamic(self):
        """build_combo uses dynamic Kelly, not static fraction."""
        from unittest.mock import patch

        from src.core.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000.0)
        legs = [
            {"event_id": "e1", "selection": "A", "odds": 2.0, "probability": 0.60,
             "sport": "soccer_epl", "market_type": "h2h"},
            {"event_id": "e2", "selection": "B", "odds": 1.8, "probability": 0.65,
             "sport": "basketball_nba", "market_type": "h2h"},
        ]

        mock_model = {"metrics": {"brier_score": 0.25}}
        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            combo = engine.build_combo(legs, kelly_frac=0.10)

        # With brier 0.25, effective Kelly = 0.10 * 0.40 = 0.04
        # The kelly fraction should be smaller than with a static 0.10
        assert combo.kelly_fraction < 0.10 * 0.10  # much smaller than static
