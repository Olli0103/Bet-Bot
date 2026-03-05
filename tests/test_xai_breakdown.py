"""Tests for XAI confidence breakdown in analyst_agent.py."""
import pytest

from src.agents.analyst_agent import AnalystAgent
from src.models.compliance import ConfidenceBreakdown


class TestBuildConfidenceBreakdown:
    def test_minimal_features(self):
        """With default/empty features, should return balanced breakdown."""
        cb = AnalystAgent.build_confidence_breakdown({}, 0.5)
        assert isinstance(cb, ConfidenceBreakdown)
        assert abs(cb.statistical_weight + cb.market_signal_weight + cb.qualitative_weight - 1.0) < 0.02

    def test_elo_dominated(self):
        """Strong Elo diff should boost statistical weight."""
        features = {
            "elo_diff": 150,
            "form_winrate_l5": 0.8,
            "team_attack_strength": 1.5,
            "opp_defense_strength": 0.7,
        }
        cb = AnalystAgent.build_confidence_breakdown(features, 0.65)
        assert cb.statistical_weight > cb.market_signal_weight
        assert any("Elo" in f for f in cb.top_factors)

    def test_steam_move_boosts_market(self):
        """Steam move trigger should boost market signal weight."""
        features = {
            "sharp_implied_prob": 0.45,
            "line_velocity": 0.03,
            "public_bias": 0.05,
        }
        cb = AnalystAgent.build_confidence_breakdown(features, 0.55, trigger="steam_move")
        assert cb.market_signal_weight > 0
        assert any("Steam" in f for f in cb.top_factors)

    def test_injury_boosts_qualitative(self):
        """Large injury delta should boost qualitative weight."""
        features = {
            "injury_delta": 5,
            "sentiment_delta": 1.0,
        }
        cb = AnalystAgent.build_confidence_breakdown(features, 0.55)
        assert cb.qualitative_weight > 0
        assert any("Verletzung" in f for f in cb.top_factors)

    def test_weights_sum_to_one(self):
        """Weights should always sum to ~1.0."""
        features = {
            "elo_diff": 80,
            "sharp_implied_prob": 0.50,
            "injury_delta": 3,
            "sentiment_delta": 0.8,
            "form_winrate_l5": 0.7,
            "market_momentum": 0.02,
        }
        cb = AnalystAgent.build_confidence_breakdown(features, 0.60, trigger="steam_move")
        total = cb.statistical_weight + cb.market_signal_weight + cb.qualitative_weight
        assert abs(total - 1.0) < 0.02

    def test_top_factors_limited(self):
        """Should never return more than 5 factors."""
        features = {
            "elo_diff": 100,
            "form_winrate_l5": 0.9,
            "team_attack_strength": 2.0,
            "opp_defense_strength": 0.5,
            "sharp_implied_prob": 0.40,
            "line_velocity": 0.05,
            "public_bias": 0.1,
            "injury_delta": 5,
            "sentiment_delta": 2.0,
            "weather_rain": 1.0,
        }
        cb = AnalystAgent.build_confidence_breakdown(features, 0.70, trigger="steam_move")
        assert len(cb.top_factors) <= 5
