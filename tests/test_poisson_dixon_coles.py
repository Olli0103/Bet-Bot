"""Tests for the Dixon-Coles adjustment in the Poisson soccer model."""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from src.core.poisson_model import PoissonSoccerModel, SCORE_RANGE


@pytest.fixture
def mock_redis():
    """Prevent Redis access during tests."""
    with patch("src.core.poisson_model.cache") as mock_cache:
        mock_cache.get_json.return_value = None
        yield mock_cache


class TestScoreMatrixDixonColes:
    """Tests for _score_matrix with Dixon-Coles rho parameter."""

    def test_rho_zero_matches_independent_poisson(self):
        """With rho=0, output should match the original independent Poisson."""
        from scipy.stats import poisson

        home_xg, away_xg = 1.5, 1.2
        matrix = PoissonSoccerModel._score_matrix(home_xg, away_xg, rho=0.0)

        home_pmf = [poisson.pmf(i, home_xg) for i in range(SCORE_RANGE)]
        away_pmf = [poisson.pmf(j, away_xg) for j in range(SCORE_RANGE)]

        for i in range(SCORE_RANGE):
            for j in range(SCORE_RANGE):
                assert abs(matrix[i][j] - home_pmf[i] * away_pmf[j]) < 1e-10

    def test_matrix_sums_to_approximately_one(self):
        """With non-zero rho, the matrix should still sum to ~1.0.

        Note: even the independent Poisson sums to < 1.0 because the
        PMF is truncated at SCORE_RANGE=7.  With Dixon-Coles normalization
        the adjusted cells are re-normalized, but the truncation tail
        remains.  We accept a tolerance of 0.01.
        """
        for rho in [-0.20, -0.13, -0.05, 0.0, 0.05]:
            matrix = PoissonSoccerModel._score_matrix(1.5, 1.2, rho=rho)
            total = sum(sum(row) for row in matrix)
            assert abs(total - 1.0) < 0.01, f"rho={rho}: matrix sums to {total}"

    def test_negative_rho_increases_draw_probability(self):
        """Negative rho (typical) should increase draw probability vs independent."""
        home_xg, away_xg = 1.4, 1.1

        matrix_indep = PoissonSoccerModel._score_matrix(home_xg, away_xg, rho=0.0)
        matrix_dc = PoissonSoccerModel._score_matrix(home_xg, away_xg, rho=-0.13)

        p_draw_indep = sum(matrix_indep[i][i] for i in range(SCORE_RANGE))
        p_draw_dc = sum(matrix_dc[i][i] for i in range(SCORE_RANGE))

        assert p_draw_dc > p_draw_indep, (
            f"Draw prob should increase with negative rho: "
            f"{p_draw_dc:.6f} vs {p_draw_indep:.6f}"
        )

    def test_negative_rho_increases_nil_nil(self):
        """P(0-0) should increase with negative rho."""
        home_xg, away_xg = 1.3, 1.0

        matrix_indep = PoissonSoccerModel._score_matrix(home_xg, away_xg, rho=0.0)
        matrix_dc = PoissonSoccerModel._score_matrix(home_xg, away_xg, rho=-0.13)

        assert matrix_dc[0][0] > matrix_indep[0][0]

    def test_all_probabilities_non_negative(self):
        """No cell in the matrix should be negative."""
        for rho in [-0.25, -0.13, 0.0, 0.10]:
            for home_xg in [0.5, 1.0, 1.5, 2.5]:
                for away_xg in [0.5, 1.0, 1.5, 2.5]:
                    matrix = PoissonSoccerModel._score_matrix(home_xg, away_xg, rho=rho)
                    for i in range(SCORE_RANGE):
                        for j in range(SCORE_RANGE):
                            assert matrix[i][j] >= 0, (
                                f"Negative prob at ({i},{j}) with "
                                f"home_xg={home_xg}, away_xg={away_xg}, rho={rho}"
                            )

    def test_predict_match_uses_rho(self, mock_redis):
        """predict_match should use the configured rho parameter."""
        model_indep = PoissonSoccerModel(rho=0.0)
        model_dc = PoissonSoccerModel(rho=-0.13)

        result_indep = model_indep.predict_match("TeamA", "TeamB")
        result_dc = model_dc.predict_match("TeamA", "TeamB")

        # Draw probability should differ
        assert result_dc["h2h_draw"] != result_indep["h2h_draw"]
        # All probabilities should be valid
        assert 0 < result_dc["h2h_home"] < 1
        assert 0 < result_dc["h2h_draw"] < 1
        assert 0 < result_dc["h2h_away"] < 1
        assert abs(result_dc["h2h_home"] + result_dc["h2h_draw"] + result_dc["h2h_away"] - 1.0) < 0.01

    def test_extreme_rho_still_valid(self):
        """Even with extreme rho, the matrix should remain valid."""
        matrix = PoissonSoccerModel._score_matrix(1.5, 1.2, rho=-0.30)
        total = sum(sum(row) for row in matrix)
        assert abs(total - 1.0) < 0.01
        for i in range(SCORE_RANGE):
            for j in range(SCORE_RANGE):
                assert matrix[i][j] >= 0
