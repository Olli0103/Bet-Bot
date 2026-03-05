"""Tests for src/core/portfolio_sizing.py — covariance, simultaneous Kelly, and CVaR."""
import numpy as np
import pytest

from src.core.portfolio_sizing import (
    _ledoit_wolf_shrinkage,
    build_covariance_matrix,
    compute_cvar,
    cvar_constrained_kelly_sizing,
    simultaneous_kelly_sizing,
)


class TestBuildCovarianceMatrix:
    def test_independent_sports(self):
        cov = build_covariance_matrix(
            n_bets=2,
            event_ids=["ev1", "ev2"],
            markets=["h2h", "h2h"],
            sports=["soccer", "basketball"],
        )
        assert cov[0, 1] == pytest.approx(0.0)
        assert cov[0, 0] == pytest.approx(0.25)

    def test_same_event_different_market(self):
        cov = build_covariance_matrix(
            n_bets=2,
            event_ids=["ev1", "ev1"],
            markets=["h2h", "totals"],
            sports=["soccer", "soccer"],
        )
        assert cov[0, 1] == pytest.approx(0.30 * 0.25)

    def test_same_sport_different_event(self):
        cov = build_covariance_matrix(
            n_bets=2,
            event_ids=["ev1", "ev2"],
            markets=["h2h", "h2h"],
            sports=["soccer", "soccer"],
        )
        assert cov[0, 1] == pytest.approx(0.05 * 0.25)

    def test_same_event_same_market(self):
        cov = build_covariance_matrix(
            n_bets=2,
            event_ids=["ev1", "ev1"],
            markets=["h2h", "h2h"],
            sports=["soccer", "soccer"],
        )
        assert cov[0, 1] == pytest.approx(0.95 * 0.25)


class TestLedoitWolfShrinkage:
    def test_shrinkage_returns_valid_matrix(self):
        np.random.seed(42)
        n_obs, n_features = 50, 3
        residuals = np.random.randn(n_obs, n_features)
        prior_cov = np.eye(n_features) * 0.25

        result = _ledoit_wolf_shrinkage(residuals, prior_cov)
        # Should be symmetric
        np.testing.assert_allclose(result, result.T, atol=1e-10)
        # Should be positive semi-definite
        eigvals = np.linalg.eigvalsh(result)
        assert eigvals.min() >= -1e-10

    def test_identity_target(self):
        np.random.seed(42)
        n_obs, n_features = 100, 3
        residuals = np.random.randn(n_obs, n_features)
        prior_cov = np.eye(n_features) * 0.25

        result = _ledoit_wolf_shrinkage(residuals, prior_cov, target="identity")
        assert result.shape == (n_features, n_features)
        np.testing.assert_allclose(result, result.T, atol=1e-10)

    def test_shrinkage_with_correlated_data(self):
        """Highly correlated data should produce off-diagonal covariance."""
        np.random.seed(42)
        n_obs = 100
        x = np.random.randn(n_obs)
        # y is highly correlated with x
        y = x + np.random.randn(n_obs) * 0.1
        residuals = np.column_stack([x, y])
        prior_cov = np.eye(2) * 0.25

        result = _ledoit_wolf_shrinkage(residuals, prior_cov)
        # Off-diagonal should be significantly positive
        assert result[0, 1] > 0.1


class TestBuildCovarianceWithResiduals:
    def test_uses_empirical_when_enough_data(self):
        np.random.seed(42)
        n_bets = 2
        residuals = np.random.randn(20, n_bets)  # 20 > 2*2 = 4

        cov_heuristic = build_covariance_matrix(
            n_bets=n_bets,
            event_ids=["ev1", "ev2"],
            markets=["h2h", "h2h"],
            sports=["soccer", "soccer"],
        )
        cov_empirical = build_covariance_matrix(
            n_bets=n_bets,
            event_ids=["ev1", "ev2"],
            markets=["h2h", "h2h"],
            sports=["soccer", "soccer"],
            historical_residuals=residuals,
        )
        # Results should differ when historical data is provided
        assert not np.allclose(cov_heuristic, cov_empirical)

    def test_falls_back_when_too_few_observations(self):
        n_bets = 3
        # Only 2 observations < 3*2 = 6 minimum
        residuals = np.random.randn(2, n_bets)

        cov_heuristic = build_covariance_matrix(
            n_bets=n_bets,
            event_ids=["ev1", "ev2", "ev3"],
            markets=["h2h", "h2h", "h2h"],
            sports=["soccer", "soccer", "soccer"],
        )
        cov_with_residuals = build_covariance_matrix(
            n_bets=n_bets,
            event_ids=["ev1", "ev2", "ev3"],
            markets=["h2h", "h2h", "h2h"],
            sports=["soccer", "soccer", "soccer"],
            historical_residuals=residuals,
        )
        # Should fall back to heuristic
        np.testing.assert_allclose(cov_heuristic, cov_with_residuals)


class TestSimultaneousKellySizing:
    def test_positive_edges(self):
        edges = np.array([0.05, 0.03])
        cov = build_covariance_matrix(
            n_bets=2,
            event_ids=["ev1", "ev2"],
            markets=["h2h", "h2h"],
            sports=["soccer", "basketball"],
        )
        fractions = simultaneous_kelly_sizing(edges, cov)
        assert all(f >= 0 for f in fractions)
        assert sum(fractions) <= 0.15 + 1e-6  # respect max exposure

    def test_empty_input(self):
        result = simultaneous_kelly_sizing(np.array([]), np.array([]))
        assert len(result) == 0


class TestComputeCVaR:
    def test_cvar_negative_for_risky_returns(self):
        """CVaR should be negative when portfolio has losses in the tail."""
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.02, 0.05, 10_000)
        cvar = compute_cvar(returns, alpha=0.05)
        assert cvar < 0.0

    def test_cvar_zero_for_all_positive(self):
        """CVaR should be positive when all returns are positive."""
        returns = np.ones(1000) * 0.01
        cvar = compute_cvar(returns, alpha=0.05)
        assert cvar > 0.0

    def test_cvar_empty_returns(self):
        assert compute_cvar(np.array([]), alpha=0.05) == 0.0

    def test_cvar_worse_than_var(self):
        """CVaR (expected shortfall) should be <= VaR for the same alpha."""
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.05, 10_000)
        alpha = 0.05
        var = np.percentile(returns, alpha * 100)
        cvar = compute_cvar(returns, alpha)
        assert cvar <= var + 1e-6


class TestCVaRConstrainedKelly:
    def _make_portfolio(self, n_bets=4):
        """Create a portfolio with correlated Bundesliga bets."""
        edges = np.array([0.08, 0.06, 0.04, 0.03])[:n_bets]
        probs = np.array([0.55, 0.52, 0.50, 0.48])[:n_bets]
        odds = np.array([2.1, 2.3, 2.5, 2.8])[:n_bets]
        cov = build_covariance_matrix(
            n_bets=n_bets,
            event_ids=[f"ev{i}" for i in range(n_bets)],
            markets=["h2h"] * n_bets,
            sports=["soccer"] * n_bets,  # same sport = correlated
        )
        return edges, cov, probs, odds

    def test_respects_exposure_cap(self):
        edges, cov, probs, odds = self._make_portfolio()
        fractions = cvar_constrained_kelly_sizing(
            edges, cov, probs, odds, max_total_exposure=0.15,
        )
        assert fractions.sum() <= 0.15 + 1e-6

    def test_all_fractions_non_negative(self):
        edges, cov, probs, odds = self._make_portfolio()
        fractions = cvar_constrained_kelly_sizing(edges, cov, probs, odds)
        assert all(f >= 0 for f in fractions)

    def test_cvar_constraint_reduces_exposure(self):
        """A tight CVaR limit should produce smaller fractions than unconstrained."""
        edges, cov, probs, odds = self._make_portfolio()

        unconstrained = simultaneous_kelly_sizing(edges, cov)
        constrained = cvar_constrained_kelly_sizing(
            edges, cov, probs, odds,
            max_cvar_loss=0.02,  # very tight: max 2% tail loss
        )
        # Constrained should be <= unconstrained (possibly equal if already safe)
        assert constrained.sum() <= unconstrained.sum() + 1e-6

    def test_loose_cvar_matches_unconstrained(self):
        """A very loose CVaR limit should not reduce the portfolio."""
        edges, cov, probs, odds = self._make_portfolio()

        unconstrained = simultaneous_kelly_sizing(edges, cov)
        constrained = cvar_constrained_kelly_sizing(
            edges, cov, probs, odds,
            max_cvar_loss=0.50,  # very loose: 50% tail loss allowed
        )
        np.testing.assert_allclose(constrained, unconstrained, atol=1e-4)

    def test_empty_input(self):
        result = cvar_constrained_kelly_sizing(
            np.array([]), np.array([]).reshape(0, 0),
            np.array([]), np.array([]),
        )
        assert len(result) == 0

    def test_single_bet_cvar(self):
        """Single bet should work without matrix issues."""
        edges = np.array([0.05])
        cov = np.array([[0.25]])
        probs = np.array([0.55])
        odds = np.array([2.1])

        fractions = cvar_constrained_kelly_sizing(edges, cov, probs, odds)
        assert len(fractions) == 1
        assert fractions[0] >= 0
