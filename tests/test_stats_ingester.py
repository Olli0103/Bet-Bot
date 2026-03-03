"""Tests for src/core/stats_ingester.py — stats ingestion and feature computation."""
import pytest

from src.core.stats_ingester import _linear_slope, _empty_snapshot


class TestLinearSlope:
    def test_ascending_values(self):
        # 0, 1, 2, 3, 4 → positive slope ≈ 1.0
        slope = _linear_slope([0, 1, 2, 3, 4])
        assert slope == pytest.approx(1.0, abs=0.01)

    def test_descending_values(self):
        # 4, 3, 2, 1, 0 → negative slope ≈ -1.0
        slope = _linear_slope([4, 3, 2, 1, 0])
        assert slope == pytest.approx(-1.0, abs=0.01)

    def test_constant_values(self):
        slope = _linear_slope([3, 3, 3, 3, 3])
        assert slope == pytest.approx(0.0, abs=0.01)

    def test_insufficient_data(self):
        assert _linear_slope([1, 2]) == 0.0
        assert _linear_slope([]) == 0.0

    def test_form_improving(self):
        # L, L, D, W, W → 0, 0, 1, 3, 3 → positive slope
        slope = _linear_slope([0, 0, 1, 3, 3])
        assert slope > 0

    def test_form_declining(self):
        # W, W, D, L, L → 3, 3, 1, 0, 0 → negative slope
        slope = _linear_slope([3, 3, 1, 0, 0])
        assert slope < 0


class TestEmptySnapshot:
    def test_returns_defaults(self):
        snap = _empty_snapshot("TestTeam", "soccer_epl", True)
        assert snap["team"] == "TestTeam"
        assert snap["sport"] == "soccer_epl"
        assert snap["is_home"] is True
        assert snap["matches_played"] == 0
        assert snap["attack_strength"] == 1.0
        assert snap["defense_strength"] == 1.0
        assert snap["form_trend_slope"] == 0.0
        assert snap["rest_days"] is None
        assert snap["schedule_congestion"] == 0.0

    def test_neutral_values(self):
        snap = _empty_snapshot("Team", "basketball_nba", False)
        assert snap["over25_rate"] == 0.0
        assert snap["btts_rate"] == 0.0
        assert snap["wins"] == 0
        assert snap["losses"] == 0
