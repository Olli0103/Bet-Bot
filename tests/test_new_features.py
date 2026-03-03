"""Tests for the Phase 4 features in feature_engineering.py."""
import pytest

from src.core.feature_engineering import FeatureEngineer


class TestPhase4Features:
    """Test the new stats-based features added in Phase 4."""

    def test_attack_defense_strength_in_output(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9, "B": 2.1},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            team_attack_strength=1.5,
            team_defense_strength=0.8,
            opp_attack_strength=1.2,
            opp_defense_strength=1.1,
        )
        assert f["team_attack_strength"] == 1.5
        assert f["team_defense_strength"] == 0.8
        assert f["opp_attack_strength"] == 1.2
        assert f["opp_defense_strength"] == 1.1

    def test_expected_total_proxy(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            team_attack_strength=1.5,
            team_defense_strength=0.8,
            opp_attack_strength=1.2,
            opp_defense_strength=1.1,
        )
        # expected = 1.5 * 1.1 * 1.35 + 1.2 * 0.8 * 1.35
        expected = 1.5 * 1.1 * 1.35 + 1.2 * 0.8 * 1.35
        assert f["expected_total_proxy"] == pytest.approx(expected, abs=0.01)

    def test_rest_fatigue_score_short_rest(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            rest_days=2,
        )
        assert f["rest_fatigue_score"] == 1.0  # very fatigued

    def test_rest_fatigue_score_medium_rest(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            rest_days=3,
        )
        assert f["rest_fatigue_score"] == 0.5

    def test_rest_fatigue_score_well_rested(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            rest_days=7,
        )
        assert f["rest_fatigue_score"] == 0.0  # well rested

    def test_rest_fatigue_score_rusty(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            rest_days=14,
        )
        assert f["rest_fatigue_score"] == 0.3  # rustiness

    def test_rest_fatigue_score_none(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            rest_days=None,
        )
        assert f["rest_fatigue_score"] == 0.0

    def test_form_trend_slope(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            form_trend_slope=0.75,
        )
        assert f["form_trend_slope"] == 0.75

    def test_schedule_congestion(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            schedule_congestion=0.35,
        )
        assert f["schedule_congestion"] == 0.35

    def test_over25_btts_rates(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            over25_rate=0.65,
            btts_rate=0.55,
        )
        assert f["over25_rate"] == 0.65
        assert f["btts_rate"] == 0.55

    def test_home_away_split_and_league_position(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            home_away_split_delta=0.15,
            league_position_delta=8.0,
        )
        assert f["home_away_split_delta"] == 0.15
        assert f["league_position_delta"] == 8.0

    def test_goals_averages(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            goals_scored_avg=2.1,
            goals_conceded_avg=0.8,
        )
        assert f["goals_scored_avg"] == 2.1
        assert f["goals_conceded_avg"] == 0.8

    def test_default_values_neutral(self):
        """Default values should not bias the model."""
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=2.0,
            sharp_market={"A": 2.0},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
        )
        assert f["team_attack_strength"] == 1.0
        assert f["team_defense_strength"] == 1.0
        assert f["form_trend_slope"] == 0.0
        assert f["rest_fatigue_score"] == 0.0
        assert f["schedule_congestion"] == 0.0

    def test_market_momentum_in_output(self):
        f = FeatureEngineer.build_core_features(
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9},
            sentiment_home=0, sentiment_away=0,
            injuries_home=0, injuries_away=0,
            selection="A", home_team="A",
            market_momentum=0.05,
        )
        assert f["market_momentum"] == 0.05
