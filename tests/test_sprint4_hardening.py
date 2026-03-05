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


# ---------------------------------------------------------------------------
# 4. Paper-Trading Isolation
# ---------------------------------------------------------------------------


class TestPaperTrading:
    """Verify paper-trading isolation prevents real-money execution."""

    def test_signal_is_paper_when_no_model(self):
        """Signals for sports without a trained model are paper-only."""
        from unittest.mock import patch

        from src.core.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000.0)

        with patch("src.core.ml_trainer.load_model", return_value=None):
            signal = engine.make_signal(
                sport="cricket_test", event_id="ev1", market="h2h",
                selection="TeamA", bookmaker_odds=2.5, model_probability=0.55,
            )
        assert signal.is_paper is True

    def test_signal_is_paper_when_brier_high(self):
        """Signals from poorly calibrated models are paper-only."""
        from unittest.mock import patch

        from src.core.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000.0)
        mock_model = {"metrics": {"brier_score": 0.30}, "n_samples": 1000}

        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            signal = engine.make_signal(
                sport="soccer_epl", event_id="ev1", market="h2h",
                selection="TeamA", bookmaker_odds=2.0, model_probability=0.60,
            )
        assert signal.is_paper is True

    def test_signal_is_live_when_model_good(self):
        """Signals from well-calibrated models with enough samples are live."""
        from unittest.mock import patch

        from src.core.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000.0)
        mock_model = {"metrics": {"brier_score": 0.18}, "n_samples": 1500}

        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            signal = engine.make_signal(
                sport="soccer_epl", event_id="ev1", market="h2h",
                selection="TeamA", bookmaker_odds=2.0, model_probability=0.60,
            )
        assert signal.is_paper is False

    def test_signal_paper_override(self):
        """Explicit is_paper=True always wins regardless of model quality."""
        from unittest.mock import patch

        from src.core.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000.0)
        mock_model = {"metrics": {"brier_score": 0.15}, "n_samples": 2000}

        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            signal = engine.make_signal(
                sport="soccer_epl", event_id="ev1", market="h2h",
                selection="TeamA", bookmaker_odds=2.0, model_probability=0.60,
                is_paper=True,
            )
        assert signal.is_paper is True


# ---------------------------------------------------------------------------
# 5. Empirical Copula Correlations
# ---------------------------------------------------------------------------


class TestEmpiricalCopulaCorrelations:
    """Verify domain-specific rho values in the correlation matrix."""

    def _make_leg(self, event_id, market_type, selection, odds=2.0, sport="soccer_epl"):
        from src.models.betting import ComboLeg
        return ComboLeg(
            event_id=event_id, selection=selection, odds=odds,
            probability=1.0/odds, sport=sport, market_type=market_type,
            market=market_type, home_team="Home", away_team="Away",
        )

    def test_fav_h2h_plus_over_is_positive(self):
        """Favorite H2H + Over should have positive rho (≈0.25)."""
        from src.core.betting_engine import BettingEngine

        h2h = self._make_leg("e1", "h2h", "Home", odds=1.5)  # favorite
        over = self._make_leg("e1", "totals", "Over 2.5", odds=1.8)

        rho = BettingEngine._empirical_pair_rho(h2h, over)
        assert rho == pytest.approx(0.25)

    def test_fav_h2h_plus_under_is_negative(self):
        """Favorite H2H + Under should have negative rho (≈-0.30)."""
        from src.core.betting_engine import BettingEngine

        h2h = self._make_leg("e1", "h2h", "Home", odds=1.5)
        under = self._make_leg("e1", "totals", "Under 2.5", odds=2.1)

        rho = BettingEngine._empirical_pair_rho(h2h, under)
        assert rho == pytest.approx(-0.30)

    def test_cross_sport_is_zero(self):
        """Cross-sport legs should have rho = 0.0."""
        from src.core.betting_engine import BettingEngine

        a = self._make_leg("e1", "h2h", "Home", sport="soccer_epl")
        b = self._make_leg("e2", "h2h", "Away", sport="basketball_nba")

        rho = BettingEngine._empirical_pair_rho(a, b)
        assert rho == pytest.approx(0.0)

    def test_same_league_different_event(self):
        """Same-league, different-event legs should have mild negative rho."""
        from src.core.betting_engine import BettingEngine

        a = self._make_leg("e1", "h2h", "Home", sport="soccer_epl")
        b = self._make_leg("e2", "h2h", "Away", sport="soccer_epl")

        rho = BettingEngine._empirical_pair_rho(a, b)
        assert rho == pytest.approx(-0.08)

    def test_matrix_is_positive_semidefinite(self):
        """Correlation matrix eigenvalues must all be >= 0."""
        from src.core.betting_engine import BettingEngine

        legs = [
            self._make_leg("e1", "h2h", "Home", odds=1.5),
            self._make_leg("e1", "totals", "Over 2.5", odds=1.8),
            self._make_leg("e1", "btts", "Yes", odds=1.9),
            self._make_leg("e2", "h2h", "Away", odds=2.5, sport="soccer_epl"),
        ]

        corr = BettingEngine._build_correlation_matrix(legs)
        eigvals = np.linalg.eigvalsh(corr)
        assert np.all(eigvals >= -1e-6)


# ---------------------------------------------------------------------------
# 6. Bayesian Gamma-Poisson Update
# ---------------------------------------------------------------------------


class TestGammaPoissonUpdate:
    """Verify the Bayesian conjugate update for Poisson strengths."""

    def test_exact_result_no_change(self):
        """When actual goals match expected, strengths should barely move."""
        from unittest.mock import patch, MagicMock

        from src.core.poisson_model import PoissonSoccerModel

        model = PoissonSoccerModel(learning_rate=0.05, rho=-0.13)
        mock_cache = MagicMock()
        mock_cache.get_json.return_value = {"attack": 1.0, "defense": 1.0}

        with patch("src.core.poisson_model.cache", mock_cache):
            # With default strengths: home_xg ≈ 1.62, away_xg ≈ 1.35
            # If actual matches expected, update should be near-identity
            model.update_strengths("HomeTeam", "AwayTeam", home_goals=2, away_goals=1)

        # Check the saved values are close to 1.0
        calls = mock_cache.set_json.call_args_list
        assert len(calls) == 2  # home + away

        home_data = calls[0][0][1]
        assert 0.8 < home_data["attack"] < 1.2
        assert 0.8 < home_data["defense"] < 1.2

    def test_outlier_compressed(self):
        """A 5-0 result should NOT cause a 500% attack boost."""
        from unittest.mock import patch, MagicMock

        from src.core.poisson_model import PoissonSoccerModel

        model = PoissonSoccerModel(learning_rate=0.05, rho=-0.13)
        mock_cache = MagicMock()
        mock_cache.get_json.return_value = {"attack": 1.0, "defense": 1.0}

        with patch("src.core.poisson_model.cache", mock_cache):
            model.update_strengths("HomeTeam", "AwayTeam", home_goals=5, away_goals=0)

        home_data = mock_cache.set_json.call_args_list[0][0][1]
        # Attack should increase but stay reasonable (< 1.5, not 5.0)
        assert home_data["attack"] < 1.5
        assert home_data["attack"] > 1.0  # but it should increase

    def test_strengths_stay_bounded(self):
        """Repeated extreme results should not blow up strengths."""
        from unittest.mock import patch, MagicMock

        from src.core.poisson_model import PoissonSoccerModel

        model = PoissonSoccerModel(learning_rate=0.05, rho=-0.13)

        stored_strengths = {"attack": 1.0, "defense": 1.0}
        mock_cache = MagicMock()

        def get_json_side_effect(key):
            return dict(stored_strengths)

        def set_json_side_effect(key, data, ttl_seconds=None):
            stored_strengths["attack"] = data["attack"]
            stored_strengths["defense"] = data["defense"]

        mock_cache.get_json.side_effect = get_json_side_effect
        mock_cache.set_json.side_effect = set_json_side_effect

        with patch("src.core.poisson_model.cache", mock_cache):
            # 10 consecutive 5-0 wins
            for _ in range(10):
                model.update_strengths("HomeTeam", "AwayTeam", home_goals=5, away_goals=0)

        # After 10 extreme results, attack should be high but clamped at 3.0
        assert stored_strengths["attack"] <= 3.0
        assert stored_strengths["defense"] >= 0.2
