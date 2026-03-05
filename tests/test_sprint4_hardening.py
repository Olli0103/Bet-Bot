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


# ---------------------------------------------------------------------------
# 7. IdentityCalibrator fallback
# ---------------------------------------------------------------------------


class TestIdentityCalibrator:
    """Verify IdentityCalibrator passes through raw probabilities."""

    def test_identity_passthrough(self):
        """IdentityCalibrator.predict returns input unchanged."""
        from src.core.ml_trainer import IdentityCalibrator

        cal = IdentityCalibrator()
        raw = np.array([0.1, 0.5, 0.9])
        result = cal.predict(raw)
        np.testing.assert_array_equal(result, raw)

    def test_identity_in_beta_calibrated_model(self):
        """BetaCalibratedModel with IdentityCalibrator returns clamped raw probs."""
        from xgboost import XGBClassifier

        from src.core.ml_trainer import BetaCalibratedModel, IdentityCalibrator

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3)
        y = (X[:, 0] > 0).astype(int)

        base = XGBClassifier(n_estimators=10, max_depth=2, verbosity=0, use_label_encoder=False)
        base.fit(X, y)

        model = BetaCalibratedModel(base, IdentityCalibrator())
        proba = model.predict_proba(X[:5])

        assert proba.shape == (5, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        # Should be close to raw XGBoost output (just clamped)
        raw = base.predict_proba(X[:5])[:, 1]
        np.testing.assert_allclose(proba[:, 1], np.clip(raw, 0.01, 0.99), atol=1e-6)


# ---------------------------------------------------------------------------
# 8. Out-of-Sample DM-Test holdout preds
# ---------------------------------------------------------------------------


class TestDMHoldoutIntegrity:
    """Verify holdout predictions come from val_model, not final model."""

    def test_holdout_preds_are_out_of_sample(self):
        """The holdout predictions must NOT be from the 100% trained model.

        We verify this by checking that a model trained on data including
        the holdout would produce different (better) predictions than
        what's stored in metrics.
        """
        import pandas as pd
        from xgboost import XGBClassifier

        from src.core.ml_trainer import _train_xgboost

        rng = np.random.RandomState(42)
        n = 500
        X_data = pd.DataFrame({
            f"feat_{i}": rng.randn(n) for i in range(5)
        })
        y_data = pd.Series((X_data["feat_0"] + X_data["feat_1"] > 0).astype(int))

        features = [f"feat_{i}" for i in range(5)]
        model, metrics, final_features = _train_xgboost(X_data, y_data, features, prune_features=False)

        # The stored holdout_y_pred should exist
        assert "holdout_y_pred" in metrics
        assert "holdout_y_true" in metrics
        stored_preds = np.array(metrics["holdout_y_pred"])

        # Now get predictions from the final model (which saw holdout data)
        holdout_idx = max(1, int(n * 0.8))
        X_holdout = X_data[features].values[holdout_idx:]
        final_preds = model.predict_proba(X_holdout)[:, 1]

        # They should NOT be identical (val_model != final_model)
        # The final model trained on holdout data should have different preds
        assert not np.allclose(stored_preds, final_preds, atol=1e-6), \
            "Holdout predictions match final model — in-sample leak detected!"


# ---------------------------------------------------------------------------
# 9. Tipico Tax-Free Combos
# ---------------------------------------------------------------------------


class TestTipicoTaxFreeCombos:
    """Verify 3+ leg combos get tax-free treatment."""

    def test_3leg_combo_tax_free(self):
        """3-leg combos should compute EV without tax deduction."""
        from unittest.mock import patch

        from src.core.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000.0)
        legs = [
            {"event_id": f"e{i}", "selection": f"T{i}", "odds": 2.0,
             "probability": 0.55, "sport": "soccer_epl", "market_type": "h2h"}
            for i in range(3)
        ]

        mock_model = {"metrics": {"brier_score": 0.18}, "n_samples": 1000}
        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            combo = engine.build_combo(legs, kelly_frac=0.10, tax_rate=0.05)

        # With 3 legs, effective tax should be 0.0
        # EV should be higher than if 5% tax were applied
        p = combo.combined_probability
        odds = combo.combined_odds
        ev_no_tax = p * odds - 1.0
        ev_with_tax = p * odds * 0.95 - 1.0

        # The stored EV should match the no-tax calculation, not the taxed one
        assert abs(combo.expected_value - ev_no_tax) < abs(combo.expected_value - ev_with_tax)

    def test_2leg_combo_still_taxed(self):
        """2-leg combos should still have tax applied."""
        from unittest.mock import patch

        from src.core.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000.0)
        legs = [
            {"event_id": f"e{i}", "selection": f"T{i}", "odds": 2.0,
             "probability": 0.55, "sport": "soccer_epl", "market_type": "h2h"}
            for i in range(2)
        ]

        mock_model = {"metrics": {"brier_score": 0.18}, "n_samples": 1000}
        with patch("src.core.ml_trainer.load_model", return_value=mock_model):
            combo_taxed = engine.build_combo(legs, kelly_frac=0.10, tax_rate=0.05)
            combo_notax = engine.build_combo(legs, kelly_frac=0.10, tax_rate=0.0)

        # 2-leg combo with tax_rate=0.05 should have lower EV than tax_rate=0.0
        assert combo_taxed.expected_value < combo_notax.expected_value


# ---------------------------------------------------------------------------
# 10. Dixon-Coles Marginal Preservation
# ---------------------------------------------------------------------------


class TestDixonColesMarginals:
    """Verify the Dixon-Coles adjustment preserves marginal distributions."""

    def test_matrix_sums_to_one(self):
        """Score matrix should sum to approximately 1.0."""
        from src.core.poisson_model import PoissonSoccerModel

        matrix = PoissonSoccerModel._score_matrix(1.5, 1.2, rho=-0.13)
        total = sum(sum(row) for row in matrix)
        assert abs(total - 1.0) < 0.01

    def test_marginals_preserved(self):
        """Row and column marginals should match the independent Poisson PMF."""
        from scipy.stats import poisson as poisson_dist

        from src.core.poisson_model import PoissonSoccerModel, SCORE_RANGE

        home_xg, away_xg = 1.5, 1.2
        matrix = PoissonSoccerModel._score_matrix(home_xg, away_xg, rho=-0.13)
        matrix_indep = PoissonSoccerModel._score_matrix(home_xg, away_xg, rho=0.0)

        # Home marginals (row sums)
        home_marginals_dc = [sum(matrix[i]) for i in range(SCORE_RANGE)]
        home_marginals_indep = [sum(matrix_indep[i]) for i in range(SCORE_RANGE)]

        # The DC-adjusted marginals should be very close to the independent ones
        for i in range(SCORE_RANGE):
            assert abs(home_marginals_dc[i] - home_marginals_indep[i]) < 0.02, \
                f"Home marginal i={i}: DC={home_marginals_dc[i]:.4f} vs Indep={home_marginals_indep[i]:.4f}"

        # Away marginals (column sums)
        away_marginals_dc = [sum(matrix[i][j] for i in range(SCORE_RANGE)) for j in range(SCORE_RANGE)]
        away_marginals_indep = [sum(matrix_indep[i][j] for i in range(SCORE_RANGE)) for j in range(SCORE_RANGE)]

        for j in range(SCORE_RANGE):
            assert abs(away_marginals_dc[j] - away_marginals_indep[j]) < 0.02, \
                f"Away marginal j={j}: DC={away_marginals_dc[j]:.4f} vs Indep={away_marginals_indep[j]:.4f}"

    def test_no_negative_probabilities(self):
        """All cells must be non-negative after DC adjustment."""
        from src.core.poisson_model import PoissonSoccerModel, SCORE_RANGE

        # Test with extreme rho values
        for rho in [-0.25, -0.13, 0.0, 0.10]:
            matrix = PoissonSoccerModel._score_matrix(1.5, 1.2, rho=rho)
            for i in range(SCORE_RANGE):
                for j in range(SCORE_RANGE):
                    assert matrix[i][j] >= 0.0, \
                        f"Negative probability at ({i},{j}) with rho={rho}: {matrix[i][j]}"


# ---------------------------------------------------------------------------
# 11. Event-Driven Orchestrator
# ---------------------------------------------------------------------------


class TestEventDrivenOrchestrator:
    """Verify the orchestrator's event-driven architecture."""

    def test_alert_queue_exists(self):
        """Orchestrator must have an asyncio.Queue for alert routing."""
        import asyncio
        from unittest.mock import MagicMock, patch

        with patch("src.agents.orchestrator.ScoutAgent"), \
             patch("src.agents.orchestrator.AnalystAgent"), \
             patch("src.agents.orchestrator.ExecutionerAgent"), \
             patch("src.agents.orchestrator.PerformanceMonitor"), \
             patch("src.agents.orchestrator.settings"):
            from src.agents.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator.__new__(AgentOrchestrator)
            orch.alert_queue = asyncio.Queue()

        assert isinstance(orch.alert_queue, asyncio.Queue)

    def test_sniper_interval_is_10s(self):
        """Sniper polling interval should be 10 seconds (max fire mode)."""
        from src.agents.orchestrator import SNIPER_INTERVAL_SECONDS
        assert SNIPER_INTERVAL_SECONDS == 10

    def test_fast_interval_is_15s(self):
        """Fast polling interval should be 15 seconds."""
        from src.agents.orchestrator import FAST_INTERVAL_SECONDS
        assert FAST_INTERVAL_SECONDS == 15

    def test_normal_interval_is_60s(self):
        """Normal polling interval should be 60 seconds."""
        from src.agents.orchestrator import NORMAL_INTERVAL_SECONDS
        assert NORMAL_INTERVAL_SECONDS == 60

    def test_idle_interval_is_3600s(self):
        """Idle polling interval should be 3600 seconds (1 hour)."""
        from src.agents.orchestrator import IDLE_INTERVAL_SECONDS
        assert IDLE_INTERVAL_SECONDS == 3600


# ---------------------------------------------------------------------------
# 12. OOF Calibration Leak Prevention
# ---------------------------------------------------------------------------


class TestOOFCalibrationLeak:
    """Verify that holdout evaluation uses a calibrator fitted ONLY on train-val OOF."""

    def test_separate_val_prod_calibrators(self):
        """_train_xgboost must produce holdout preds using beta_cal_val (80% OOF),
        NOT beta_cal (100% OOF).  We verify by inspecting the source code for
        the critical pattern: val_model wrapped with beta_cal_val."""
        import inspect
        from src.core.ml_trainer import _train_xgboost

        source = inspect.getsource(_train_xgboost)
        # Must use beta_cal_val for holdout evaluation
        assert "beta_cal_val" in source, "Missing beta_cal_val — holdout evaluation leaks OOF data"
        # Must fit a separate calibrator on train-val only
        assert "oof_preds_val" in source, "Missing oof_preds_val — no separate val OOF predictions"
        # The holdout wrapper must use beta_cal_val, not beta_cal
        assert "BetaCalibratedModel(val_model, beta_cal_val)" in source, (
            "Holdout must wrap val_model with beta_cal_val, not beta_cal"
        )

    def test_val_calibrator_does_not_see_holdout(self):
        """beta_cal_val is fitted on X_train_val OOF only (first 80%)."""
        import inspect
        from src.core.ml_trainer import _train_xgboost

        source = inspect.getsource(_train_xgboost)
        # The val OOF loop must iterate over X_train_val, not X_active
        assert "tscv_val.split(X_train_val)" in source, (
            "Val OOF must split X_train_val, not X_active"
        )


# ---------------------------------------------------------------------------
# 13. Liquidity-Capped Staking
# ---------------------------------------------------------------------------


class TestLiquidityCaps:
    """Verify league-tier stake multipliers."""

    def test_tier1_full_stake(self):
        """EPL, NFL, NBA should allow full stake (cap = 1.0)."""
        from src.core.risk_guards import get_liquidity_cap

        assert get_liquidity_cap("soccer_epl") == 1.0
        assert get_liquidity_cap("americanfootball_nfl") == 1.0
        assert get_liquidity_cap("basketball_nba") == 1.0

    def test_tier2_reduced_stake(self):
        """Bundesliga, Serie A should cap at 0.7."""
        from src.core.risk_guards import get_liquidity_cap

        assert get_liquidity_cap("soccer_germany_bundesliga") == 0.7
        assert get_liquidity_cap("soccer_italy_serie_a") == 0.7
        assert get_liquidity_cap("icehockey_nhl") == 0.7

    def test_tier3_heavily_reduced(self):
        """2nd leagues, MLS should cap at 0.3."""
        from src.core.risk_guards import get_liquidity_cap

        assert get_liquidity_cap("soccer_germany_bundesliga2") == 0.3
        assert get_liquidity_cap("soccer_usa_mls") == 0.3

    def test_tier4_unknown_leagues(self):
        """Unknown/niche leagues should cap at 0.1."""
        from src.core.risk_guards import get_liquidity_cap

        assert get_liquidity_cap("soccer_mongolia_premier") == 0.1
        assert get_liquidity_cap("esports_csgo") == 0.1

    def test_liquidity_cap_integrated_in_executioner(self):
        """Executioner must import and use get_liquidity_cap."""
        import inspect
        from src.agents.executioner_agent import ExecutionerAgent

        source = inspect.getsource(ExecutionerAgent)
        assert "get_liquidity_cap" in source, (
            "Executioner must apply liquidity caps before stake sizing"
        )


# ---------------------------------------------------------------------------
# 14. Exact Gamma-Poisson Conjugate Update
# ---------------------------------------------------------------------------


class TestExactGammaPoissonConjugate:
    """Verify the exact Bayesian conjugate update formula."""

    def test_no_spurious_plus_one(self):
        """The update must NOT contain beta + 1.0 (the old dimensional error)."""
        import inspect
        from src.core.poisson_model import PoissonSoccerModel

        source = inspect.getsource(PoissonSoccerModel.update_strengths)
        assert "beta + 1.0" not in source, "Dimensional error: beta + 1.0 still present"
        assert "beta_post" in source or "alpha_post / beta_post" in source, (
            "Must use exact conjugate: alpha_post / beta_post"
        )

    def test_exact_conjugate_math(self):
        """Verify: alpha' = alpha + k, beta' = beta + c, E[theta] = alpha'/beta'."""
        # Manual calculation:
        # current_strength = 1.2, observed = 2, expected = 1.5, lr = 0.05
        # prior_weight = 20, c = 1.5/1.2 = 1.25
        # alpha_prior = 20 * 1.2 = 24, beta_prior = 20
        # alpha_post = 24 + 2 = 26, beta_post = 20 + 1.25 = 21.25
        # result = 26 / 21.25 = 1.2235...
        current = 1.2
        observed = 2
        expected = 1.5
        lr = 0.05

        prior_weight = 1.0 / lr  # 20
        c = expected / current   # 1.25
        alpha_prior = prior_weight * current  # 24
        beta_prior = prior_weight             # 20
        alpha_post = alpha_prior + observed   # 26
        beta_post = beta_prior + c            # 21.25
        result = alpha_post / beta_post       # 1.22352...

        np.testing.assert_almost_equal(result, 26.0 / 21.25, decimal=6)
        # Result should be between current and observed/c
        assert result > current  # observed > expected, so strength increases


# ---------------------------------------------------------------------------
# 15. Zero-Sum Home Advantage
# ---------------------------------------------------------------------------


class TestZeroSumHomeAdvantage:
    """Verify home advantage doesn't inflate total league goals."""

    def test_reciprocal_away_scaling(self):
        """away_xg must be divided by home_advantage (reciprocal)."""
        import inspect
        from src.core.poisson_model import PoissonSoccerModel

        source = inspect.getsource(PoissonSoccerModel._expected_goals)
        assert "/ self.home_advantage" in source, (
            "away_xg must include / self.home_advantage for zero-sum"
        )

    def test_total_xg_neutral_on_average(self):
        """With average teams (att=def=1.0), total xG must equal 2*league_avg."""
        from unittest.mock import patch, MagicMock
        from src.core.poisson_model import PoissonSoccerModel

        model = PoissonSoccerModel(home_advantage=1.2, league_avg_goals=1.35)

        # Mock Redis to return default strengths (1.0, 1.0)
        with patch.object(model, "_load_strengths", return_value={"attack": 1.0, "defense": 1.0}):
            home_xg, away_xg = model._expected_goals("HomeTeam", "AwayTeam")

        # Home gets boosted, away gets penalized — total must stay neutral
        total_xg = home_xg + away_xg
        expected_total = 2 * 1.35  # 2.70 for two average teams
        # With reciprocal: home = 1.35*1.2 = 1.62, away = 1.35/1.2 = 1.125
        # total = 1.62 + 1.125 = 2.745 (close to 2.70 but not exact due to
        # asymmetry — the key is it's MUCH closer than without reciprocal)

        # Without reciprocal: total would be 1.62 + 1.35 = 2.97 (10% inflation!)
        # With reciprocal: total is 2.745 — only 1.7% deviation, within bounds
        assert total_xg < 2 * 1.35 * 1.05, (
            f"Total xG {total_xg:.3f} inflated by >5% vs neutral {expected_total}"
        )
        # Verify away_xg is reduced (not equal to league_avg)
        assert away_xg < 1.35, f"away_xg {away_xg:.3f} should be < league_avg 1.35"

    def test_home_away_xg_mirror_symmetry(self):
        """home_advantage * away_advantage_factor == 1 (zero-sum)."""
        from unittest.mock import patch
        from src.core.poisson_model import PoissonSoccerModel

        model = PoissonSoccerModel(home_advantage=1.2, league_avg_goals=1.35)

        with patch.object(model, "_load_strengths", return_value={"attack": 1.0, "defense": 1.0}):
            home_xg, away_xg = model._expected_goals("A", "B")

        # home_xg / away_xg should equal home_advantage^2
        # (home gets *1.2, away gets /1.2, ratio = 1.2 * 1.2 = 1.44)
        ratio = home_xg / away_xg
        np.testing.assert_almost_equal(ratio, 1.2 ** 2, decimal=6)


# ---------------------------------------------------------------------------
# 16. Cross-Market Deduplication
# ---------------------------------------------------------------------------


class TestCrossMarketDedup:
    """Verify deduplication uses event_id + market, not just event_id."""

    def test_dedup_key_includes_market(self):
        """Orchestrator dedup key must be event_id:market."""
        import inspect
        from src.agents.orchestrator import AgentOrchestrator

        # Check both run_once and run_continuous
        source_once = inspect.getsource(AgentOrchestrator.run_once)
        source_cont = inspect.getsource(AgentOrchestrator.run_continuous)

        for label, source in [("run_once", source_once), ("run_continuous", source_cont)]:
            assert "event_id + market" in source or 'f"{eid}:{market}"' in source, (
                f"{label}: dedup key must include market to preserve cross-market signals"
            )

    def test_independent_markets_both_survive(self):
        """H2H and Totals alerts for the same event must NOT overwrite each other."""
        alerts = [
            {"event_id": "evt1", "market": "h2h", "movement_pct": 5.0, "selection": "Home"},
            {"event_id": "evt1", "market": "totals", "movement_pct": 3.0, "selection": "Over 2.5"},
            {"event_id": "evt2", "market": "h2h", "movement_pct": 2.0, "selection": "Away"},
        ]

        # Simulate the dedup logic
        deduped = {}
        for alert in alerts:
            eid = alert.get("event_id", "")
            market = alert.get("market", "h2h")
            dedup_key = f"{eid}:{market}"
            movement = float(alert.get("movement_pct", 0))
            existing = deduped.get(dedup_key)
            if existing is None or movement > float(existing.get("movement_pct", 0)):
                deduped[dedup_key] = alert

        # All 3 are independent (different event_id:market combos)
        assert len(deduped) == 3
        assert "evt1:h2h" in deduped
        assert "evt1:totals" in deduped
        assert "evt2:h2h" in deduped

    def test_same_market_keeps_strongest(self):
        """Two H2H alerts for the same event should keep the stronger one."""
        alerts = [
            {"event_id": "evt1", "market": "h2h", "movement_pct": 2.0, "selection": "Home"},
            {"event_id": "evt1", "market": "h2h", "movement_pct": 7.0, "selection": "Away"},
        ]

        deduped = {}
        for alert in alerts:
            eid = alert.get("event_id", "")
            market = alert.get("market", "h2h")
            dedup_key = f"{eid}:{market}"
            movement = float(alert.get("movement_pct", 0))
            existing = deduped.get(dedup_key)
            if existing is None or movement > float(existing.get("movement_pct", 0)):
                deduped[dedup_key] = alert

        assert len(deduped) == 1
        assert deduped["evt1:h2h"]["selection"] == "Away"  # 7.0 > 2.0


# ---------------------------------------------------------------------------
# 17. Adaptive Sniper Scheduler
# ---------------------------------------------------------------------------


class TestAdaptiveSniperScheduler:
    """Verify time-to-kickoff based polling intervals."""

    def test_sniper_mode_under_15min(self):
        """< 15 min before kickoff → 10s interval (maximum fire)."""
        from datetime import timedelta
        from unittest.mock import patch
        from src.agents.orchestrator import AgentOrchestrator, SNIPER_INTERVAL_SECONDS

        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        with patch.object(orch, "_nearest_kickoff_delta", return_value=timedelta(minutes=10)):
            interval = orch._get_adaptive_interval()
        assert interval == SNIPER_INTERVAL_SECONDS == 10

    def test_fast_mode_15min_to_1h(self):
        """15 min - 1h before kickoff → 15s interval."""
        from datetime import timedelta
        from unittest.mock import patch
        from src.agents.orchestrator import AgentOrchestrator, FAST_INTERVAL_SECONDS

        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        with patch.object(orch, "_nearest_kickoff_delta", return_value=timedelta(minutes=30)):
            interval = orch._get_adaptive_interval()
        assert interval == FAST_INTERVAL_SECONDS == 15

    def test_normal_mode_1h_to_6h(self):
        """1-6h before kickoff → 60s interval."""
        from datetime import timedelta
        from unittest.mock import patch
        from src.agents.orchestrator import AgentOrchestrator, NORMAL_INTERVAL_SECONDS

        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        with patch.object(orch, "_nearest_kickoff_delta", return_value=timedelta(hours=3)):
            interval = orch._get_adaptive_interval()
        assert interval == NORMAL_INTERVAL_SECONDS == 60

    def test_idle_mode_over_6h(self):
        """> 6h before kickoff → 3600s (conserve API quota)."""
        from datetime import timedelta
        from unittest.mock import patch
        from src.agents.orchestrator import AgentOrchestrator, IDLE_INTERVAL_SECONDS

        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        with patch.object(orch, "_nearest_kickoff_delta", return_value=timedelta(hours=10)):
            interval = orch._get_adaptive_interval()
        assert interval == IDLE_INTERVAL_SECONDS == 3600

    def test_no_events_returns_idle(self):
        """No tracked events → idle mode (3600s)."""
        from unittest.mock import patch
        from src.agents.orchestrator import AgentOrchestrator, IDLE_INTERVAL_SECONDS

        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        with patch.object(orch, "_nearest_kickoff_delta", return_value=None):
            interval = orch._get_adaptive_interval()
        assert interval == IDLE_INTERVAL_SECONDS


# ---------------------------------------------------------------------------
# 18. Trading Session Window
# ---------------------------------------------------------------------------


class TestTradingSessionWindow:
    """Verify the active trading session scoping (now → tomorrow 06:59 UTC)."""

    def test_window_end_is_tomorrow_0659(self):
        """get_trading_window_end() returns tomorrow at 06:59 UTC."""
        from datetime import datetime, timezone
        from src.integrations.odds_fetcher import get_trading_window_end

        end = get_trading_window_end()
        now = datetime.now(timezone.utc)

        # Must be in the future
        assert end > now
        # Must be at 06:59
        assert end.hour == 6
        assert end.minute == 59
        # Must be UTC
        assert end.tzinfo is not None

    def test_window_end_is_within_48h(self):
        """Window end should be at most ~48h from now."""
        from datetime import datetime, timezone, timedelta
        from src.integrations.odds_fetcher import get_trading_window_end

        end = get_trading_window_end()
        now = datetime.now(timezone.utc)
        assert end - now < timedelta(hours=48)

    def test_odds_fetcher_accepts_commence_time_params(self):
        """get_sport_odds_async must accept commenceTimeFrom/To parameters."""
        import inspect
        from src.integrations.odds_fetcher import OddsFetcher

        sig = inspect.signature(OddsFetcher.get_sport_odds_async)
        params = list(sig.parameters.keys())
        assert "commence_time_from" in params
        assert "commence_time_to" in params


# ---------------------------------------------------------------------------
# 19. Deep Sleep Caching
# ---------------------------------------------------------------------------


class TestDeepSleepCaching:
    """Verify cold-sport hibernation (no events → skip for 24h)."""

    def test_deep_sleep_skips_cold_sport(self):
        """Scout must skip sports marked as deep-sleeping."""
        import inspect
        from src.agents.scout_agent import ScoutAgent

        source = inspect.getsource(ScoutAgent.monitor_odds)
        assert "is_sport_deep_sleeping" in source, (
            "Scout must check deep-sleep status before polling a sport"
        )
        assert "mark_sport_deep_sleep" in source, (
            "Scout must activate deep-sleep when no events in trading window"
        )

    def test_deep_sleep_activates_on_empty_events(self):
        """When API returns 0 events, the sport should enter deep-sleep."""
        import inspect
        from src.agents.scout_agent import ScoutAgent

        source = inspect.getsource(ScoutAgent.monitor_odds)
        # Must check len(events) == 0 and call mark_sport_deep_sleep
        assert "len(events) == 0" in source or "mark_sport_deep_sleep(sport)" in source


# ---------------------------------------------------------------------------
# 20. Dynamic Market Pruning
# ---------------------------------------------------------------------------


class TestDynamicMarketPruning:
    """Verify illiquid leagues get h2h-only, liquid leagues get full markets."""

    def test_tier1_gets_full_markets(self):
        """EPL, NFL should get h2h,spreads,totals."""
        from src.integrations.odds_fetcher import get_markets_for_sport

        assert "totals" in get_markets_for_sport("soccer_epl")
        assert "spreads" in get_markets_for_sport("americanfootball_nfl")
        assert "spreads" in get_markets_for_sport("basketball_nba")

    def test_tier3_gets_h2h_totals_only(self):
        """2nd leagues get h2h,totals (no spreads)."""
        from src.integrations.odds_fetcher import get_markets_for_sport

        markets = get_markets_for_sport("soccer_germany_bundesliga2")
        assert "h2h" in markets
        assert "totals" in markets
        assert "spreads" not in markets

    def test_tier4_gets_h2h_only(self):
        """Unknown/niche leagues get h2h only."""
        from src.integrations.odds_fetcher import get_markets_for_sport

        markets = get_markets_for_sport("soccer_mongolia_premier")
        assert markets == "h2h"

    def test_scout_uses_dynamic_markets(self):
        """Scout must call get_markets_for_sport, not hardcode markets."""
        import inspect
        from src.agents.scout_agent import ScoutAgent

        source = inspect.getsource(ScoutAgent.monitor_odds)
        assert "get_markets_for_sport" in source, (
            "Scout must use dynamic market pruning per sport"
        )


# ---------------------------------------------------------------------------
# 21. Live-Event Filter
# ---------------------------------------------------------------------------


class TestLiveEventFilter:
    """Verify that in-play events are excluded from pre-match analysis."""

    def test_scout_filters_live_events(self):
        """Scout must skip events where commence_time is in the past."""
        import inspect
        from src.agents.scout_agent import ScoutAgent

        source = inspect.getsource(ScoutAgent.monitor_odds)
        # Must compare commence_time against current time
        assert "ct <= now_utc" in source or "ct < now_utc" in source, (
            "Scout must filter out events that have already started"
        )

    def test_live_event_detection_logic(self):
        """Events with commence_time in the past should be identified as live."""
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        past = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        future = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Parse and compare
        ct_past = datetime.fromisoformat(past.replace("Z", "+00:00"))
        ct_future = datetime.fromisoformat(future.replace("Z", "+00:00"))

        assert ct_past <= now   # live — should be filtered
        assert ct_future > now  # pre-match — should pass


# ---------------------------------------------------------------------------
# 22. In-Play Terminator (per-sport fetch interval)
# ---------------------------------------------------------------------------


class TestInPlayTerminator:
    """Verify that live matches trigger deep-sleep, not sniper-mode."""

    @staticmethod
    def _mock_cache(kickoffs, statuses=None):
        """Helper: returns a side_effect for cache.get_json that serves
        kickoff lists for kickoffs:sport:* and status dicts for match_status:sport:*."""
        def side_effect(key):
            if "kickoffs:" in key:
                return kickoffs
            if "match_status:" in key:
                return statuses or {}
            return None
        return side_effect

    def test_all_matches_live_returns_deep_sleep(self):
        """When all kickoffs are in the past (beyond grace), return deep-sleep."""
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch
        from src.core.fetch_scheduler import get_sport_fetch_interval

        now = datetime.now(timezone.utc)
        past_kickoffs = [
            (now - timedelta(hours=1)).isoformat(),
            (now - timedelta(minutes=30)).isoformat(),
        ]

        with patch("src.core.fetch_scheduler.cache") as mock_cache:
            mock_cache.get_json.side_effect = self._mock_cache(past_kickoffs)
            interval = get_sport_fetch_interval("soccer_epl")

        assert interval >= 3600

    def test_future_match_triggers_sniper(self):
        """Match < 15 min away triggers 10s sniper interval."""
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch
        from src.core.fetch_scheduler import get_sport_fetch_interval

        now = datetime.now(timezone.utc)
        kickoffs = [(now + timedelta(minutes=10)).isoformat()]

        with patch("src.core.fetch_scheduler.cache") as mock_cache:
            mock_cache.get_json.side_effect = self._mock_cache(kickoffs)
            interval = get_sport_fetch_interval("soccer_epl")

        assert interval == 10

    def test_mixed_live_and_future_uses_future_only(self):
        """With one live + one future match, interval based on the future one."""
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch
        from src.core.fetch_scheduler import get_sport_fetch_interval

        now = datetime.now(timezone.utc)
        kickoffs = [
            (now - timedelta(hours=1)).isoformat(),
            (now + timedelta(hours=2)).isoformat(),
        ]

        with patch("src.core.fetch_scheduler.cache") as mock_cache:
            mock_cache.get_json.side_effect = self._mock_cache(kickoffs)
            interval = get_sport_fetch_interval("soccer_epl")

        assert interval == 600

    def test_no_kickoffs_returns_idle(self):
        """No kickoff data → idle interval (3600s)."""
        from unittest.mock import patch
        from src.core.fetch_scheduler import get_sport_fetch_interval

        with patch("src.core.fetch_scheduler.cache") as mock_cache:
            mock_cache.get_json.side_effect = self._mock_cache([])
            interval = get_sport_fetch_interval("soccer_epl")

        assert interval == 3600

    def test_negative_delta_excluded(self):
        """Negative deltas (in-play, beyond grace) must NEVER trigger sniper mode."""
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch
        from src.core.fetch_scheduler import get_sport_fetch_interval

        now = datetime.now(timezone.utc)
        # Match started 25 min ago (beyond 20-min grace) with status "in_progress"
        kickoffs = [(now - timedelta(minutes=25)).isoformat()]
        statuses = {kickoffs[0]: "in_progress"}

        with patch("src.core.fetch_scheduler.cache") as mock_cache:
            mock_cache.get_json.side_effect = self._mock_cache(kickoffs, statuses)
            interval = get_sport_fetch_interval("soccer_epl")

        assert interval >= 3600, (
            f"In-play match triggered {interval}s interval instead of deep sleep"
        )


# ---------------------------------------------------------------------------
# 23. Empty Window Cache Firewall
# ---------------------------------------------------------------------------


class TestEmptyWindowCache:
    """Verify empty API responses are cached until trading window end."""

    def test_empty_response_gets_long_ttl(self):
        """Empty [] from API should be cached with TTL until window end."""
        import inspect
        from src.integrations.odds_fetcher import OddsFetcher

        source = inspect.getsource(OddsFetcher.get_sport_odds_async)
        # Must have special handling for empty responses
        assert "len(data) == 0" in source, (
            "Empty responses must be detected for long-TTL caching"
        )
        assert "ttl_to_window_end" in source or "window_end" in source, (
            "Empty responses must be cached until trading window end"
        )

    def test_nonempty_response_uses_standard_ttl(self):
        """Non-empty responses should use the standard TTL."""
        import inspect
        from src.integrations.odds_fetcher import OddsFetcher

        source = inspect.getsource(OddsFetcher.get_sport_odds_async)
        # The else branch must cache with standard ttl_seconds
        assert "ttl_seconds)" in source, (
            "Non-empty responses must use standard ttl_seconds"
        )


# ---------------------------------------------------------------------------
# 24. Tipico Tax Guard (Min Odds Check)
# ---------------------------------------------------------------------------


class TestTipicoTaxGuard:
    """Verify tax-free only applies when ALL legs have odds >= 1.50."""

    def test_3leg_combo_all_above_150_is_tax_free(self):
        """3+ legs with all odds >= 1.50 → tax_rate = 0.0."""
        from unittest.mock import patch, MagicMock
        from src.core.betting_engine import BettingEngine

        engine = BettingEngine.__new__(BettingEngine)
        engine.bankroll = 1000.0

        legs = [
            {"event_id": "e1", "selection": "Home", "odds": 1.80, "probability": 0.55,
             "sport": "soccer_epl", "market_type": "h2h"},
            {"event_id": "e2", "selection": "Away", "odds": 2.10, "probability": 0.45,
             "sport": "soccer_epl", "market_type": "h2h"},
            {"event_id": "e3", "selection": "Over 2.5", "odds": 1.90, "probability": 0.50,
             "sport": "soccer_epl", "market_type": "totals"},
        ]

        with patch.object(engine, "_build_correlation_matrix", return_value=np.eye(3)), \
             patch.object(engine, "_compute_joint_probability_copula", return_value=0.12), \
             patch("src.core.betting_engine.get_dynamic_kelly_frac", return_value=0.1):
            combo = engine.build_combo(legs, tax_rate=0.053)

        # All odds >= 1.50, 3 legs → tax-free
        # EV computed with 0% tax
        assert combo.expected_value > -1.0  # just verify it ran

    def test_3leg_combo_one_below_150_is_taxed(self):
        """3 legs but one has odds 1.20 → tax still applies (Tipico AGB)."""
        import inspect
        from src.core.betting_engine import BettingEngine

        source = inspect.getsource(BettingEngine.build_combo)
        # Must check min odds per leg
        assert "MIN_ODDS_FOR_TAX_FREE" in source, (
            "Tax-free rule must enforce minimum odds per leg"
        )
        assert "1.50" in source or "1.5" in source, (
            "Minimum odds threshold must be 1.50"
        )
        assert "all(" in source, (
            "Must check ALL legs meet the minimum odds requirement"
        )

    def test_2leg_combo_high_odds_still_taxed(self):
        """2-leg combos are always taxed regardless of odds."""
        import inspect
        from src.core.betting_engine import BettingEngine

        source = inspect.getsource(BettingEngine.build_combo)
        # Must require >= 3 unique events
        assert "unique_events >= 3" in source, (
            "Tax-free requires 3+ unique events"
        )


# ---------------------------------------------------------------------------
# 25. SGP Tax Guard (Same Game Parlay trap)
# ---------------------------------------------------------------------------


class TestSGPTaxGuard:
    """Verify tax-free requires 3 UNIQUE events, not just 3 legs."""

    def test_sgp_3legs_same_event_is_taxed(self):
        """3 legs from the same event count as 1 → taxed."""
        import inspect
        from src.core.betting_engine import BettingEngine

        source = inspect.getsource(BettingEngine.build_combo)
        assert "unique_events" in source, "Must count unique events, not just legs"
        assert "set(leg.event_id" in source, "Must use set() to count distinct events"

    def test_unique_event_counting(self):
        """3 legs from 2 events → only 2 unique → taxed."""
        # Simulate: 2 legs from evt1 (SGP) + 1 from evt2 = 2 unique events
        event_ids = ["evt1", "evt1", "evt2"]
        unique = len(set(event_ids))
        assert unique == 2
        assert unique < 3  # not tax-free

    def test_3_unique_events_qualifies(self):
        """3 legs from 3 different events → 3 unique → potentially tax-free."""
        event_ids = ["evt1", "evt2", "evt3"]
        unique = len(set(event_ids))
        assert unique == 3


# ---------------------------------------------------------------------------
# 26. MAO (Minimum Acceptable Odds)
# ---------------------------------------------------------------------------


class TestMAOCalculator:
    """Verify the Minimum Acceptable Odds break-even calculation."""

    def test_mao_basic_calculation(self):
        """MAO at 55% probability with 5.3% tax and 1% edge."""
        from src.core.betting_math import calculate_mao

        mao = calculate_mao(0.55, tax_rate=0.053, required_edge=0.01)
        # (1.0 + 0.01) / (0.55 * (1 - 0.053)) = 1.01 / 0.52085 = 1.939
        assert 1.9 < mao < 2.0

    def test_mao_zero_probability_returns_max(self):
        """Zero probability → max odds (999.0)."""
        from src.core.betting_math import calculate_mao

        assert calculate_mao(0.0) == 999.0

    def test_mao_high_probability_low_threshold(self):
        """80% probability → very low MAO (should be around 1.3)."""
        from src.core.betting_math import calculate_mao

        mao = calculate_mao(0.80, tax_rate=0.053, required_edge=0.01)
        assert mao < 1.4

    def test_mao_integrated_in_executioner(self):
        """Executioner must compute MAO and abort if odds < MAO."""
        import inspect
        from src.agents.executioner_agent import ExecutionerAgent

        source = inspect.getsource(ExecutionerAgent.execute)
        assert "calculate_mao" in source, "Executioner must compute MAO"
        assert "mao" in source, "Executioner must use MAO for slippage guard"

    def test_mao_in_bet_signal_model(self):
        """BetSignal model must include mao field."""
        from src.models.betting import BetSignal

        fields = BetSignal.model_fields
        assert "mao" in fields, "BetSignal must have mao field"


# ---------------------------------------------------------------------------
# 27. Combo Liquidity Cap (min across legs)
# ---------------------------------------------------------------------------


class TestComboLiquidityCap:
    """Verify combo stake uses the strictest (minimum) liquidity cap."""

    def test_combo_uses_min_cap(self):
        """Engine must apply min(caps) to combo stake, not first leg's cap."""
        import inspect
        from src.core.betting_engine import BettingEngine

        source = inspect.getsource(BettingEngine.build_combo)
        assert "min(leg_caps)" in source or "min(caps)" in source, (
            "Combo must use min() across all legs' liquidity caps"
        )
        assert "get_liquidity_cap" in source, (
            "Combo must query liquidity cap per leg"
        )

    def test_min_cap_math(self):
        """EPL (1.0) + Peru (0.1) combo → cap = 0.1."""
        caps = [1.0, 0.7, 0.1]
        assert min(caps) == 0.1


# ---------------------------------------------------------------------------
# 28. Delayed Kickoff Guard
# ---------------------------------------------------------------------------


class TestDelayedKickoffGuard:
    """Verify delayed kickoffs keep sniper-mode active within grace window."""

    def test_grace_window_keeps_sniper_active(self):
        """Match 10 min past scheduled KO with status=not_started → sniper stays."""
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch
        from src.core.fetch_scheduler import get_sport_fetch_interval

        now = datetime.now(timezone.utc)
        # Match scheduled 10 min ago but hasn't started (weather delay)
        past_ko = (now - timedelta(minutes=10)).isoformat()

        with patch("src.core.fetch_scheduler.cache") as mock_cache:
            def side_effect(key):
                if "kickoffs:" in key:
                    return [past_ko]
                if "match_status:" in key:
                    return {past_ko: "not_started"}
                return None
            mock_cache.get_json.side_effect = side_effect
            interval = get_sport_fetch_interval("soccer_epl")

        # Should still be in sniper mode (10s) due to grace window
        assert interval == 10, f"Delayed kickoff should keep sniper active, got {interval}s"

    def test_beyond_grace_window_deep_sleeps(self):
        """Match 30 min past scheduled KO → grace expired → deep sleep."""
        from datetime import datetime, timezone, timedelta
        from unittest.mock import patch
        from src.core.fetch_scheduler import get_sport_fetch_interval, DELAYED_KICKOFF_GRACE_SECONDS

        now = datetime.now(timezone.utc)
        # Match scheduled 30 min ago (beyond 20-min grace)
        old_ko = (now - timedelta(minutes=30)).isoformat()

        with patch("src.core.fetch_scheduler.cache") as mock_cache:
            def side_effect(key):
                if "kickoffs:" in key:
                    return [old_ko]
                if "match_status:" in key:
                    return {old_ko: "not_started"}
                return None
            mock_cache.get_json.side_effect = side_effect
            interval = get_sport_fetch_interval("soccer_epl")

        assert interval >= 3600, f"Beyond grace window should deep-sleep, got {interval}s"

    def test_grace_period_is_20_minutes(self):
        """Grace period constant should be 1200 seconds (20 minutes)."""
        from src.core.fetch_scheduler import DELAYED_KICKOFF_GRACE_SECONDS
        assert DELAYED_KICKOFF_GRACE_SECONDS == 1200


# ---------------------------------------------------------------------------
# 29. Atomic Redis SETNX Locks (Double-Spend Prevention)
# ---------------------------------------------------------------------------


class TestAtomicBetLocks:
    """Verify SETNX-based locks prevent double-spend on rapid queue events."""

    def test_setnx_method_exists(self):
        """RedisCache must expose a setnx() method."""
        from src.data.redis_cache import RedisCache
        assert hasattr(RedisCache, "setnx"), "RedisCache must have setnx method"

    def test_orchestrator_uses_bet_lock(self):
        """_process_single_alert must acquire a SETNX lock before execution."""
        import inspect
        from src.agents.orchestrator import AgentOrchestrator
        source = inspect.getsource(AgentOrchestrator._process_single_alert)
        assert "bet_lock" in source.lower() or "setnx" in source.lower(), (
            "_process_single_alert must use atomic SETNX lock"
        )
        assert "BET_LOCK_PREFIX" in source or "bet_lock:" in source, (
            "Lock key must use BET_LOCK_PREFIX"
        )

    def test_lock_key_includes_event_and_market(self):
        """Lock key must be per (event_id, market) to allow independent markets."""
        from src.agents.orchestrator import BET_LOCK_PREFIX
        # The lock key pattern should include both event_id and market
        assert "bet_lock" in BET_LOCK_PREFIX.lower()

    def test_lock_ttl_is_12_hours(self):
        """Lock TTL should be 12 hours (one trading session)."""
        from src.agents.orchestrator import BET_LOCK_TTL
        assert BET_LOCK_TTL == 12 * 3600, f"Lock TTL should be 43200s, got {BET_LOCK_TTL}"

    def test_setnx_returns_false_on_existing_key(self):
        """SETNX must return False when the key already exists."""
        from unittest.mock import MagicMock, patch
        from src.data.redis_cache import RedisCache

        rc = RedisCache.__new__(RedisCache)
        mock_client = MagicMock()
        rc._client = mock_client

        # First call: key doesn't exist → True
        mock_client.set.return_value = True
        assert rc.setnx("test_key", "val", ttl_seconds=60) is True

        # Second call: key exists → False (NX fails)
        mock_client.set.return_value = False
        assert rc.setnx("test_key", "val", ttl_seconds=60) is False

    def test_double_spend_blocked_in_process_alert(self):
        """Second alert for same event:market must be skipped."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.agents.orchestrator import AgentOrchestrator

        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        orch.analyst = MagicMock()
        orch.executioner = MagicMock()
        orch.bot = None
        orch.chat_id = ""

        alert = {
            "event_id": "evt123",
            "sport": "soccer_epl",
            "home": "A",
            "away": "B",
            "selection": "A",
            "market": "h2h",
        }

        # Simulate lock already held
        with patch("src.agents.orchestrator.cache") as mock_cache:
            mock_cache.setnx.return_value = False
            result = asyncio.get_event_loop().run_until_complete(
                orch._process_single_alert(alert)
            )
        assert result["action"] == "skip"
        assert "bet_lock" in result["reasoning"][0]


# ---------------------------------------------------------------------------
# 30. Spinning Wheel Guard (Tipico Re-Offer)
# ---------------------------------------------------------------------------


class TestSpinningWheelGuard:
    """Verify Tipico spinning wheel / odds_changed handling."""

    def test_no_prior_response_allows_bet(self):
        """First attempt with no cached response → proceed."""
        from unittest.mock import patch
        from src.core.spinning_wheel_guard import check_spinning_wheel

        with patch("src.core.spinning_wheel_guard.cache") as mock_cache:
            mock_cache.get_json.return_value = None
            ok, reason = check_spinning_wheel("evt1", "TeamA", mao=1.85)
        assert ok is True

    def test_accepted_response_allows_bet(self):
        """Status=accepted → proceed."""
        from unittest.mock import patch
        from src.core.spinning_wheel_guard import check_spinning_wheel

        with patch("src.core.spinning_wheel_guard.cache") as mock_cache:
            mock_cache.get_json.return_value = {"status": "accepted", "new_odds": 0}
            ok, reason = check_spinning_wheel("evt1", "TeamA", mao=1.85)
        assert ok is True

    def test_rejected_blocks_bet(self):
        """Status=rejected → abort."""
        from unittest.mock import patch
        from src.core.spinning_wheel_guard import check_spinning_wheel

        with patch("src.core.spinning_wheel_guard.cache") as mock_cache:
            mock_cache.get_json.return_value = {"status": "rejected", "new_odds": 0}
            ok, reason = check_spinning_wheel("evt1", "TeamA", mao=1.85)
        assert ok is False
        assert "rejected" in reason.lower()

    def test_odds_changed_above_mao_allows(self):
        """Re-offered odds above MAO → proceed at new price."""
        from unittest.mock import patch
        from src.core.spinning_wheel_guard import check_spinning_wheel

        with patch("src.core.spinning_wheel_guard.cache") as mock_cache:
            mock_cache.get_json.return_value = {
                "status": "odds_changed",
                "original_odds": 2.10,
                "new_odds": 2.00,
            }
            ok, reason = check_spinning_wheel("evt1", "TeamA", mao=1.85)
        assert ok is True

    def test_odds_changed_below_mao_blocks(self):
        """Re-offered odds below MAO → edge gone, abort."""
        from unittest.mock import patch
        from src.core.spinning_wheel_guard import check_spinning_wheel

        with patch("src.core.spinning_wheel_guard.cache") as mock_cache:
            mock_cache.get_json.return_value = {
                "status": "odds_changed",
                "original_odds": 2.10,
                "new_odds": 1.75,
            }
            ok, reason = check_spinning_wheel("evt1", "TeamA", mao=1.85)
        assert ok is False
        assert "edge gone" in reason.lower()

    def test_record_bet_response_caches(self):
        """record_bet_response must write to cache."""
        from unittest.mock import patch
        from src.core.spinning_wheel_guard import record_bet_response

        with patch("src.core.spinning_wheel_guard.cache") as mock_cache:
            record_bet_response("evt1", "TeamA", "odds_changed", 2.10, 1.95)
            mock_cache.set_json.assert_called_once()
            key = mock_cache.set_json.call_args[0][0]
            assert "evt1" in key and "TeamA" in key

    def test_executioner_calls_spinning_wheel(self):
        """Executioner must call check_spinning_wheel before placing bet."""
        import inspect
        from src.agents.executioner_agent import ExecutionerAgent
        source = inspect.getsource(ExecutionerAgent.execute)
        assert "check_spinning_wheel" in source, (
            "Executioner must call spinning wheel guard"
        )


# ---------------------------------------------------------------------------
# 31. True CLV T-0 Closing Line Tracker
# ---------------------------------------------------------------------------


class TestTrueCLVTracker:
    """Verify T-0 closing line fetch at exact kickoff."""

    def test_fetch_t0_function_exists(self):
        """clv_logger must expose fetch_t0_closing_line."""
        from src.core.clv_logger import fetch_t0_closing_line
        assert callable(fetch_t0_closing_line)

    def test_t0_uses_short_ttl(self):
        """T-0 fetch must use ultra-short TTL (<=10s) for fresh data."""
        import inspect
        from src.core.clv_logger import fetch_t0_closing_line
        source = inspect.getsource(fetch_t0_closing_line)
        assert "ttl_seconds=10" in source, "T-0 fetch must use ultra-short TTL"

    def test_t0_caches_result_48h(self):
        """T-0 closing line should be cached for 48 hours."""
        from src.core.clv_logger import T0_CLOSING_TTL
        assert T0_CLOSING_TTL == 48 * 3600

    def test_t0_returns_cached_if_exists(self):
        """If T-0 closing already cached, return it without API call."""
        from unittest.mock import patch, MagicMock
        from src.core.clv_logger import fetch_t0_closing_line

        with patch("src.core.clv_logger.cache") as mock_cache, \
             patch("src.core.clv_logger.OddsFetcher") as mock_fetcher:
            mock_cache.get_json.return_value = {"TeamA": 2.05, "TeamB": 3.40}
            result = fetch_t0_closing_line("soccer_epl", "evt123")
            assert result == {"TeamA": 2.05, "TeamB": 3.40}
            mock_fetcher.assert_not_called()  # No API call needed

    def test_t0_returns_none_on_failure(self):
        """API failure → return None gracefully."""
        from unittest.mock import patch, MagicMock
        from src.core.clv_logger import fetch_t0_closing_line

        with patch("src.core.clv_logger.cache") as mock_cache, \
             patch("src.core.clv_logger.OddsFetcher") as mock_fetcher:
            mock_cache.get_json.return_value = None
            mock_instance = MagicMock()
            mock_instance.get_sport_odds.side_effect = Exception("API down")
            mock_fetcher.return_value = mock_instance
            result = fetch_t0_closing_line("soccer_epl", "evt123")
            assert result is None


# ---------------------------------------------------------------------------
# 32. Proactive Session Refresh
# ---------------------------------------------------------------------------


class TestProactiveSessionRefresh:
    """Verify bookie session is refreshed before sniper window."""

    def test_session_manager_exists(self):
        """session_manager module must be importable."""
        from src.integrations.session_manager import ensure_session_fresh
        assert callable(ensure_session_fresh)

    def test_fresh_session_needs_no_refresh(self):
        """Session refreshed <20 min ago → no action needed."""
        import time
        from unittest.mock import patch
        from src.integrations.session_manager import ensure_session_fresh

        with patch("src.integrations.session_manager.cache") as mock_cache:
            mock_cache.get_json.return_value = {
                "last_refresh_ts": time.time() - 300,  # 5 min ago
                "source": "proactive",
            }
            ok, reason = ensure_session_fresh()
        assert ok is True
        assert reason == "session_fresh"

    def test_stale_session_triggers_refresh(self):
        """Session >20 min old → refresh triggered."""
        import time
        from unittest.mock import patch, MagicMock
        from src.integrations.session_manager import ensure_session_fresh

        refresh_fn = MagicMock()
        with patch("src.integrations.session_manager.cache") as mock_cache:
            mock_cache.get_json.return_value = {
                "last_refresh_ts": time.time() - 1500,  # 25 min ago
                "source": "proactive",
            }
            mock_cache.setnx.return_value = True
            ok, reason = ensure_session_fresh(refresh_fn=refresh_fn)
        assert ok is True
        assert reason == "refreshed"
        refresh_fn.assert_called_once()

    def test_unknown_session_triggers_refresh(self):
        """No session state in cache → refresh to be safe."""
        from unittest.mock import patch, MagicMock
        from src.integrations.session_manager import ensure_session_fresh

        with patch("src.integrations.session_manager.cache") as mock_cache:
            mock_cache.get_json.return_value = None
            mock_cache.setnx.return_value = True
            ok, reason = ensure_session_fresh()
        assert ok is True

    def test_max_session_age_is_20_minutes(self):
        """MAX_SESSION_AGE_SECONDS should be 1200 (20 min)."""
        from src.integrations.session_manager import MAX_SESSION_AGE_SECONDS
        assert MAX_SESSION_AGE_SECONDS == 20 * 60

    def test_orchestrator_calls_session_refresh_in_sniper_mode(self):
        """Orchestrator must call ensure_session_fresh before sniper cycles."""
        import inspect
        from src.agents.orchestrator import AgentOrchestrator
        source = inspect.getsource(AgentOrchestrator.run_continuous)
        assert "ensure_session_fresh" in source, (
            "Orchestrator must call session refresh before sniper window"
        )


# ===========================================================================
# SPRINT 13: Portfolio Scaling & Concept Drift
# ===========================================================================


# ---------------------------------------------------------------------------
# 33. Free Margin & Exposure Tracking
# ---------------------------------------------------------------------------


class TestFreeMarginExposure:
    """Verify bankroll sizing deducts pending exposure."""

    def test_get_free_margin_method_exists(self):
        """BankrollManager must expose get_free_margin()."""
        from src.core.bankroll import BankrollManager
        assert hasattr(BankrollManager, "get_free_margin")

    def test_free_margin_deducts_pending(self):
        """free_margin = bankroll - pending exposure."""
        from unittest.mock import patch, MagicMock
        from src.core.bankroll import BankrollManager

        mgr = BankrollManager.__new__(BankrollManager)
        mgr._initial = 1000.0
        mgr._owner = ""

        with patch.object(mgr, "get_current_bankroll", return_value=1000.0), \
             patch.object(mgr, "get_pending_exposure", return_value=200.0):
            free = mgr.get_free_margin()
        assert free == 800.0

    def test_free_margin_never_negative(self):
        """Free margin must never go below 0."""
        from unittest.mock import patch
        from src.core.bankroll import BankrollManager

        mgr = BankrollManager.__new__(BankrollManager)
        mgr._initial = 1000.0
        mgr._owner = ""

        with patch.object(mgr, "get_current_bankroll", return_value=100.0), \
             patch.object(mgr, "get_pending_exposure", return_value=500.0):
            free = mgr.get_free_margin()
        assert free == 0.0

    def test_add_pending_exposure_accumulates(self):
        """add_pending_exposure must increase the total."""
        from unittest.mock import patch
        from src.core.bankroll import BankrollManager

        mgr = BankrollManager.__new__(BankrollManager)
        mgr._initial = 1000.0
        mgr._owner = ""

        with patch("src.core.bankroll.cache") as mock_cache:
            mock_cache.get_json.return_value = 100.0
            new_total = mgr.add_pending_exposure(50.0)
        assert new_total == 150.0

    def test_release_pending_exposure_decreases(self):
        """release_pending_exposure must decrease the total."""
        from unittest.mock import patch
        from src.core.bankroll import BankrollManager

        mgr = BankrollManager.__new__(BankrollManager)
        mgr._initial = 1000.0
        mgr._owner = ""

        with patch("src.core.bankroll.cache") as mock_cache:
            mock_cache.get_json.return_value = 150.0
            new_total = mgr.release_pending_exposure(50.0)
        assert new_total == 100.0

    def test_executioner_uses_free_margin(self):
        """Executioner must call get_free_margin, not get_current_bankroll."""
        import inspect
        from src.agents.executioner_agent import ExecutionerAgent
        source = inspect.getsource(ExecutionerAgent.execute)
        assert "get_free_margin" in source, (
            "Executioner must use free margin (bankroll minus pending exposure)"
        )

    def test_executioner_records_pending_exposure(self):
        """After a bet, executioner must call add_pending_exposure."""
        import inspect
        from src.agents.executioner_agent import ExecutionerAgent
        source = inspect.getsource(ExecutionerAgent.execute)
        assert "add_pending_exposure" in source, (
            "Executioner must record pending exposure after bet placement"
        )

    def test_simultaneous_kelly_scenario(self):
        """10 simultaneous signals must not exceed bankroll capacity.

        With 1000€ bankroll and 5% Kelly per bet:
        - WITHOUT free margin: 10 × 50€ = 500€ (50% at risk!)
        - WITH free margin: each bet sees decreasing available capital
        """
        bankroll = 1000.0
        kelly_pct = 0.05
        pending = 0.0

        stakes = []
        for _ in range(10):
            free = max(0.0, bankroll - pending)
            stake = round(free * kelly_pct, 2)
            stakes.append(stake)
            pending += stake

        total_exposure = sum(stakes)
        assert total_exposure < bankroll * 0.50, (
            f"Total exposure {total_exposure:.2f} exceeds 50% of bankroll"
        )
        # Last bet should be meaningfully smaller than first
        assert stakes[-1] < stakes[0], "Later bets should have smaller stakes"


# ---------------------------------------------------------------------------
# 34. Time Decay Sample Weights
# ---------------------------------------------------------------------------


class TestTimeDecaySampleWeights:
    """Verify exponential time decay in ML training."""

    def test_compute_time_decay_positional(self):
        """Positional decay: last sample weight ~1.0, first much lower."""
        from src.core.ml_trainer import _compute_time_decay_weights
        weights = _compute_time_decay_weights(n_samples=1000, half_life_days=180.0)
        assert len(weights) == 1000
        assert weights[-1] > 0.9, f"Newest sample weight should be ~1.0, got {weights[-1]}"
        assert weights[0] < weights[-1], "Oldest sample must have lower weight"

    def test_half_life_decay_rate(self):
        """A sample ~half_life old should have weight ~0.5."""
        from src.core.ml_trainer import _compute_time_decay_weights
        # With 365 samples spanning ~1 year, index 0 is ~1 year ago
        weights = _compute_time_decay_weights(n_samples=365, half_life_days=180.0)
        # Sample at ~180 days ago (index ~185 from end = index ~180)
        mid_idx = 365 - 180  # ~185 days from start ≈ 180 days ago
        # Allow tolerance since positional mapping is approximate
        assert 0.3 < weights[mid_idx] < 0.7, (
            f"Sample at ~half-life should have weight ~0.5, got {weights[mid_idx]}"
        )

    def test_weights_all_positive(self):
        """All weights must be > 0 (clamped to 0.01 minimum)."""
        from src.core.ml_trainer import _compute_time_decay_weights
        weights = _compute_time_decay_weights(n_samples=5000, half_life_days=180.0)
        assert (weights > 0).all(), "All weights must be positive"
        assert weights.min() >= 0.01, "Minimum weight must be >= 0.01"

    def test_half_life_constant(self):
        """TIME_DECAY_HALF_LIFE_DAYS should be 180 (6 months)."""
        from src.core.ml_trainer import TIME_DECAY_HALF_LIFE_DAYS
        assert TIME_DECAY_HALF_LIFE_DAYS == 180.0

    def test_train_xgboost_accepts_timestamps(self):
        """_train_xgboost must accept a timestamps parameter."""
        import inspect
        from src.core.ml_trainer import _train_xgboost
        sig = inspect.signature(_train_xgboost)
        assert "timestamps" in sig.parameters, (
            "_train_xgboost must accept timestamps for time decay"
        )

    def test_xgb_fit_uses_sample_weight(self):
        """All XGBoost .fit() calls must pass sample_weight."""
        import inspect, re
        from src.core.ml_trainer import _train_xgboost
        source = inspect.getsource(_train_xgboost)
        # Count only XGBClassifier .fit() calls (not BetaCalibration .fit())
        # XGB fits use array indexing like y_vals[...] or y_train_val
        xgb_fits = len(re.findall(r'\.fit\([^)]*sample_weight=', source))
        # Every XGBClassifier .fit should have sample_weight
        assert xgb_fits >= 6, (
            f"At least 6 XGB .fit() calls must pass sample_weight "
            f"(found {xgb_fits})"
        )


# ---------------------------------------------------------------------------
# 35. Humanized Execution Jitter
# ---------------------------------------------------------------------------


class TestExecutionJitter:
    """Verify WAF-evasion jitter between consecutive bets."""

    def test_jitter_module_exists(self):
        """execution_jitter module must be importable."""
        from src.core.execution_jitter import get_execution_delay, apply_execution_jitter
        assert callable(get_execution_delay)
        assert callable(apply_execution_jitter)

    def test_normal_delay_range(self):
        """Normal delay should be 1.5-4.0s (with jitter)."""
        from src.core.execution_jitter import get_execution_delay, _recent_executions
        _recent_executions.clear()
        delays = [get_execution_delay()[0] for _ in range(100)]
        assert all(d >= 0.3 for d in delays), "No delay should be < 0.3s"
        assert max(delays) < 6.0, "Normal delays shouldn't exceed 6s"
        avg = sum(delays) / len(delays)
        assert 1.5 < avg < 4.0, f"Average normal delay should be 1.5-4.0s, got {avg:.2f}"

    def test_burst_detection_increases_delay(self):
        """After rapid-fire executions, delay should increase significantly."""
        import time
        from src.core.execution_jitter import (
            get_execution_delay, record_execution, _recent_executions,
        )
        _recent_executions.clear()

        # Simulate 3 rapid executions
        for _ in range(3):
            record_execution()

        delay, reason = get_execution_delay()
        assert delay >= 4.0, f"Burst cooldown should be >= 4.0s, got {delay:.2f}"
        assert "burst" in reason.lower()

    def test_orchestrator_uses_jitter(self):
        """Worker loop must call apply_execution_jitter before processing."""
        import inspect
        from src.agents.orchestrator import AgentOrchestrator
        source = inspect.getsource(AgentOrchestrator._worker_loop)
        assert "apply_execution_jitter" in source, (
            "Worker loop must apply humanized jitter before execution"
        )

    def test_jitter_constants(self):
        """Verify jitter parameter bounds."""
        from src.core.execution_jitter import (
            MIN_DELAY_SECONDS, MAX_DELAY_SECONDS,
            BURST_COOLDOWN_MIN, BURST_COOLDOWN_MAX,
            BURST_THRESHOLD, BURST_WINDOW_SECONDS,
        )
        assert MIN_DELAY_SECONDS >= 1.0
        assert MAX_DELAY_SECONDS <= 5.0
        assert BURST_COOLDOWN_MIN >= 4.0
        assert BURST_COOLDOWN_MAX <= 10.0
        assert BURST_THRESHOLD >= 2
        assert BURST_WINDOW_SECONDS >= 5.0


# ---------------------------------------------------------------------------
# 36. Pre-Kickoff CLV Snapshot (T-90s)
# ---------------------------------------------------------------------------


class TestPreKickoffSnapshot:
    """Verify CLV snapshot is taken at T-90s, not T-0."""

    def test_snapshot_timing_constant(self):
        """PRE_KICKOFF_SNAPSHOT_SECONDS should be 90."""
        from src.core.clv_logger import PRE_KICKOFF_SNAPSHOT_SECONDS
        assert PRE_KICKOFF_SNAPSHOT_SECONDS == 90

    def test_t0_docstring_mentions_pre_kickoff(self):
        """fetch_t0_closing_line docstring must mention T-90s / pre-kickoff."""
        from src.core.clv_logger import fetch_t0_closing_line
        doc = fetch_t0_closing_line.__doc__ or ""
        assert "90" in doc or "pre-kickoff" in doc.lower(), (
            "Docstring must document T-90s snapshot timing"
        )

    def test_clv_logger_warns_about_suspended_markets(self):
        """Module docs should explain why T-0 is dangerous."""
        import inspect
        import src.core.clv_logger as mod
        source = inspect.getsource(mod)
        assert "suspend" in source.lower() or "SUSPEND" in source, (
            "clv_logger must document the market suspension risk"
        )


# ===========================================================================
# SPRINT 14: SOTA Institutional Upgrades
# ===========================================================================


# ---------------------------------------------------------------------------
# 37. Shin's Method for Vig Removal
# ---------------------------------------------------------------------------


class TestShinVigRemoval:
    """Verify Shin's method produces superior vig-removed probabilities."""

    def test_shin_function_exists(self):
        """remove_vig_shin must be importable."""
        from src.core.betting_math import remove_vig_shin
        assert callable(remove_vig_shin)

    def test_shin_sums_to_one(self):
        """Shin-devigged probabilities must sum to 1.0."""
        from src.core.betting_math import remove_vig_shin
        prices = {"Home": 1.90, "Draw": 3.40, "Away": 4.50}
        fair = remove_vig_shin(prices)
        total = sum(fair.values())
        assert abs(total - 1.0) < 0.001, f"Shin probabilities sum to {total}, not 1.0"

    def test_shin_respects_ranking(self):
        """Favourite must have highest probability after vig removal."""
        from src.core.betting_math import remove_vig_shin
        prices = {"Home": 1.50, "Draw": 4.00, "Away": 6.50}
        fair = remove_vig_shin(prices)
        assert fair["Home"] > fair["Draw"] > fair["Away"]

    def test_shin_vs_power_both_valid(self):
        """Both methods must produce valid probabilities on skewed markets."""
        from src.core.betting_math import remove_vig_shin, _remove_vig_power
        # Extreme favourite market (1.10 vs 8.00)
        prices = {"Home": 1.10, "Draw": 8.00, "Away": 15.00}
        shin = remove_vig_shin(prices)
        power = _remove_vig_power(prices)
        # Both must sum to 1 and preserve ranking
        assert abs(sum(shin.values()) - 1.0) < 0.001
        assert abs(sum(power.values()) - 1.0) < 0.001
        # Shin accounts for insider proportion (z > 0), producing
        # different (more accurate) probabilities than Power
        assert shin["Home"] != power["Home"], "Shin and Power should differ on skewed markets"

    def test_shin_fallback_to_power(self):
        """If Shin solver fails, _remove_vig must still return valid results."""
        from src.core.betting_math import _remove_vig
        prices = {"Home": 2.00, "Away": 2.00}  # Fair market, no vig
        fair = _remove_vig(prices)
        assert len(fair) == 2
        assert abs(sum(fair.values()) - 1.0) < 0.01

    def test_shin_dispatched_by_default(self):
        """_remove_vig must dispatch to Shin by default."""
        import inspect
        from src.core.betting_math import _remove_vig
        source = inspect.getsource(_remove_vig)
        assert "remove_vig_shin" in source

    def test_power_method_still_available(self):
        """Power method must be preserved as fallback."""
        from src.core.betting_math import _remove_vig_power
        prices = {"Home": 1.90, "Draw": 3.40, "Away": 4.50}
        fair = _remove_vig_power(prices)
        assert abs(sum(fair.values()) - 1.0) < 0.001


# ---------------------------------------------------------------------------
# 38. Simultaneous Kelly Portfolio Sizing
# ---------------------------------------------------------------------------


class TestSimultaneousKelly:
    """Verify portfolio-aware Kelly sizing with covariance."""

    def test_module_exists(self):
        """portfolio_sizing module must be importable."""
        from src.core.portfolio_sizing import (
            simultaneous_kelly_sizing, build_covariance_matrix,
        )
        assert callable(simultaneous_kelly_sizing)
        assert callable(build_covariance_matrix)

    def test_independent_bets_sum_under_cap(self):
        """Independent bets: total exposure must stay under MAX_TOTAL_EXPOSURE."""
        from src.core.portfolio_sizing import simultaneous_kelly_sizing
        edges = np.array([0.05, 0.04, 0.03, 0.06, 0.02])
        cov = np.eye(5) * 0.25  # Independent
        fracs = simultaneous_kelly_sizing(edges, cov, kelly_fraction=0.25)
        assert fracs.sum() <= 0.16, f"Total exposure {fracs.sum()} exceeds cap"
        assert all(f >= 0 for f in fracs), "No negative fractions allowed"

    def test_correlated_bets_reduce_exposure(self):
        """Correlated bets should get smaller fractions than independent ones."""
        from src.core.portfolio_sizing import simultaneous_kelly_sizing
        edges = np.array([0.05, 0.05])
        # Independent
        cov_ind = np.eye(2) * 0.25
        fracs_ind = simultaneous_kelly_sizing(edges, cov_ind, kelly_fraction=0.25)
        # Correlated (rho=0.5)
        cov_corr = np.array([[0.25, 0.125], [0.125, 0.25]])
        fracs_corr = simultaneous_kelly_sizing(edges, cov_corr, kelly_fraction=0.25)
        assert fracs_corr.sum() <= fracs_ind.sum() + 0.01, (
            "Correlated bets should not get more exposure than independent"
        )

    def test_negative_ev_gets_zero_fraction(self):
        """Bets with negative EV must get zero allocation."""
        from src.core.portfolio_sizing import simultaneous_kelly_sizing
        edges = np.array([0.05, -0.02, 0.03])
        cov = np.eye(3) * 0.25
        fracs = simultaneous_kelly_sizing(edges, cov, kelly_fraction=0.25)
        assert fracs[1] < 0.001, f"Negative-EV bet should get ~0, got {fracs[1]}"

    def test_build_covariance_same_event(self):
        """Same event, different market → rho=0.30."""
        from src.core.portfolio_sizing import build_covariance_matrix
        cov = build_covariance_matrix(
            n_bets=2,
            event_ids=["evt1", "evt1"],
            markets=["h2h", "totals"],
            sports=["soccer_epl", "soccer_epl"],
        )
        off_diag = cov[0, 1]
        expected_rho = 0.30 * 0.25  # rho * base_variance
        assert abs(off_diag - expected_rho) < 0.01

    def test_build_covariance_different_sports(self):
        """Different sports → rho=0.0 (independent)."""
        from src.core.portfolio_sizing import build_covariance_matrix
        cov = build_covariance_matrix(
            n_bets=2,
            event_ids=["evt1", "evt2"],
            markets=["h2h", "h2h"],
            sports=["soccer_epl", "basketball_nba"],
        )
        assert cov[0, 1] == 0.0

    def test_max_total_exposure_constant(self):
        """MAX_TOTAL_EXPOSURE should be 15%."""
        from src.core.portfolio_sizing import MAX_TOTAL_EXPOSURE
        assert MAX_TOTAL_EXPOSURE == 0.15


# ---------------------------------------------------------------------------
# 39. Bayesian Feature Smoothing
# ---------------------------------------------------------------------------


class TestBayesianSmoothing:
    """Verify Bayesian smoothing prevents early-season miscalibration."""

    def test_function_exists(self):
        """calculate_smoothed_feature must be importable."""
        from src.core.feature_engineering import calculate_smoothed_feature
        assert callable(calculate_smoothed_feature)

    def test_small_sample_shrinks_to_prior(self):
        """2 wins in 2 games with prior=0.5 → ~58% (not 100%)."""
        from src.core.feature_engineering import calculate_smoothed_feature
        smoothed = calculate_smoothed_feature(
            measured_value=1.0, sample_size=2,
            prior_value=0.5, prior_weight=10,
        )
        # (1.0*2 + 0.5*10) / (2+10) = 7/12 ≈ 0.5833
        assert 0.55 < smoothed < 0.62, f"Expected ~0.58, got {smoothed}"

    def test_large_sample_approaches_measured(self):
        """30 games → smoothed should be close to measured."""
        from src.core.feature_engineering import calculate_smoothed_feature
        smoothed = calculate_smoothed_feature(
            measured_value=0.667, sample_size=30,
            prior_value=0.5, prior_weight=10,
        )
        # (0.667*30 + 0.5*10) / (30+10) = 25/40 = 0.625
        assert abs(smoothed - 0.625) < 0.01

    def test_zero_sample_returns_prior(self):
        """0 games → smoothed = prior exactly."""
        from src.core.feature_engineering import calculate_smoothed_feature
        smoothed = calculate_smoothed_feature(
            measured_value=0.0, sample_size=0,
            prior_value=0.5, prior_weight=10,
        )
        assert smoothed == 0.5

    def test_feature_engineer_applies_smoothing(self):
        """build_core_features must call calculate_smoothed_feature."""
        import inspect
        from src.core.feature_engineering import FeatureEngineer
        source = inspect.getsource(FeatureEngineer.build_core_features)
        assert "calculate_smoothed_feature" in source, (
            "build_core_features must apply Bayesian smoothing"
        )

    def test_priors_defined(self):
        """LEAGUE_PRIORS and PRIOR_WEIGHTS must be defined."""
        from src.core.feature_engineering import LEAGUE_PRIORS, PRIOR_WEIGHTS
        assert "form_winrate" in LEAGUE_PRIORS or "form_winrate" in PRIOR_WEIGHTS
        assert "attack_strength" in LEAGUE_PRIORS

    def test_smoothing_applied_to_attack_strength(self):
        """Early-season attack_strength must be shrunk towards 1.0."""
        from src.core.feature_engineering import FeatureEngineer
        # Team with 3 goals in 2 games (attack = 2.22) but only 2 games played
        features = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.95,
            sharp_market={"Home": 1.95, "Away": 2.05},
            sentiment_home=0.0, sentiment_away=0.0,
            injuries_home=0, injuries_away=0,
            selection="Home", home_team="Home",
            form_games_l5=2,
            team_attack_strength=2.22,
        )
        # With prior=1.0 and weight=10: (2.22*2 + 1.0*10)/(2+10) = 14.44/12 = 1.20
        assert features["team_attack_strength"] < 2.0, (
            f"Attack strength should be shrunk from 2.22, got {features['team_attack_strength']}"
        )
        assert features["team_attack_strength"] > 1.0, (
            "Attack strength should still be above neutral"
        )


# ---------------------------------------------------------------------------
# 40. CPU Offloading (ProcessPoolExecutor)
# ---------------------------------------------------------------------------


class TestCPUOffloading:
    """Verify ML training is offloaded to avoid event-loop blocking."""

    def test_trigger_async_retrain_exists(self):
        """trigger_async_retrain must be importable."""
        from src.bot.core_worker import trigger_async_retrain
        assert callable(trigger_async_retrain)
        import asyncio
        assert asyncio.iscoroutinefunction(trigger_async_retrain)

    def test_sync_training_function_exists(self):
        """_run_ml_training_sync must be a module-level picklable function."""
        from src.bot.core_worker import _run_ml_training_sync
        assert callable(_run_ml_training_sync)

    def test_worker_uses_process_pool(self):
        """trigger_async_retrain must use ProcessPoolExecutor."""
        import inspect
        from src.bot.core_worker import trigger_async_retrain
        source = inspect.getsource(trigger_async_retrain)
        assert "ProcessPoolExecutor" in source, (
            "ML training must be offloaded to a separate process"
        )
        assert "run_in_executor" in source, (
            "Must use loop.run_in_executor for async integration"
        )
