"""Tests for src/agents/analyst_agent.py — probability pipeline and injury logic.

Mocks external dependencies (Redis, ML models, sentiment, etc.) to test
the analysis logic in isolation.
"""
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from src.agents.analyst_agent import AnalystAgent


@pytest.fixture
def analyst():
    with patch("src.agents.analyst_agent.QuantPricingModel") as MockQPM, \
         patch("src.agents.analyst_agent.EloSystem") as MockElo, \
         patch("src.agents.analyst_agent.PoissonSoccerModel") as MockPoisson:
        # Configure QuantPricingModel mock
        qpm_inst = MockQPM.return_value
        qpm_inst.get_true_probability.return_value = 0.55

        # Configure EloSystem mock
        elo_inst = MockElo.return_value
        elo_inst.get_elo_features.return_value = {"elo_diff": 30.0, "elo_expected": 0.6}

        # Configure Poisson mock (not used for non-soccer)
        poisson_inst = MockPoisson.return_value
        poisson_inst.predict_match.return_value = {
            "h2h_home": 0.45, "h2h_draw": 0.28, "h2h_away": 0.27,
            "home_xg": 1.5, "away_xg": 1.2,
        }

        agent = AnalystAgent()
        yield agent


@pytest.fixture
def mock_externals():
    """Patch all external calls the analyst makes."""
    with patch("src.agents.analyst_agent.team_sentiment_score", return_value=0.3), \
         patch("src.agents.analyst_agent.get_form_l5", return_value=(0.6, 5)), \
         patch("src.agents.analyst_agent.get_h2h_features", return_value={"h2h_home_winrate": 0.55}), \
         patch("src.agents.analyst_agent.get_volatility_features", return_value={"home_volatility": 0.02, "away_volatility": 0.03}):
        # Also mock injury aggregator (imported inside the method)
        with patch.dict("sys.modules", {
            "src.integrations.injury_aggregator": MagicMock(
                aggregate_injury_intel=AsyncMock(return_value={"injuries": []}),
                get_injury_impact_score=MagicMock(return_value=0.0),
            ),
        }):
            yield


class TestAnalyzeEventBasic:
    @pytest.mark.asyncio
    async def test_returns_required_fields(self, analyst, mock_externals):
        result = await analyst.analyze_event(
            event_id="test_001",
            sport="basketball_nba",
            home="Lakers",
            away="Celtics",
            selection="Lakers",
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"Lakers": 1.9, "Celtics": 2.1},
        )
        assert "model_probability" in result
        assert "expected_value" in result
        assert "recommendation" in result
        assert "features" in result
        assert "sentiment" in result
        assert "form" in result
        assert "elo" in result

    @pytest.mark.asyncio
    async def test_probability_is_bounded(self, analyst, mock_externals):
        result = await analyst.analyze_event(
            event_id="test_002",
            sport="basketball_nba",
            home="Lakers",
            away="Celtics",
            selection="Lakers",
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"Lakers": 1.9, "Celtics": 2.1},
        )
        p = result["model_probability"]
        assert 0.01 <= p <= 0.99

    @pytest.mark.asyncio
    async def test_preserves_input_params(self, analyst, mock_externals):
        """Verify the fix: sharp_odds, sharp_market, market_momentum are in result."""
        result = await analyst.analyze_event(
            event_id="test_003",
            sport="basketball_nba",
            home="Lakers",
            away="Celtics",
            selection="Lakers",
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"Lakers": 1.9, "Celtics": 2.1},
            market_momentum=0.05,
        )
        assert result["sharp_odds"] == 1.9
        assert result["sharp_market"] == {"Lakers": 1.9, "Celtics": 2.1}
        assert result["market_momentum"] == 0.05
        assert result["bookmaker_odds"] == 2.0


class TestInjuryPenaltySign:
    """Verify the injury penalty sign fix: injuries on selected team should DECREASE model_p."""

    @pytest.mark.asyncio
    async def test_selected_team_injury_decreases_prob(self, analyst, mock_externals):
        """When selected team (home) has injuries but opponent doesn't,
        model_p should be LOWER than without injuries."""
        # Run without injury penalty
        with patch("src.agents.analyst_agent.get_volatility_features",
                    return_value={"home_volatility": 0.0, "away_volatility": 0.0}):
            result_no_injury = await analyst.analyze_event(
                event_id="inj_test_1",
                sport="basketball_nba",
                home="Lakers",
                away="Celtics",
                selection="Lakers",
                target_odds=2.0,
                sharp_odds=1.9,
                sharp_market={"Lakers": 1.9, "Celtics": 2.1},
            )

        # Now with injury penalty on selected team
        def mock_injury_aggregator():
            m = MagicMock()
            m.aggregate_injury_intel = AsyncMock(return_value={
                "injuries": [{"player": "LeBron", "team": "Lakers", "status": "Out"}]
            })
            # Negative = injury hurts this team
            m.get_injury_impact_score = MagicMock(side_effect=lambda injuries, team:
                -0.15 if "lakers" in team.lower() else 0.0
            )
            return m

        with patch.dict("sys.modules", {
            "src.integrations.injury_aggregator": mock_injury_aggregator(),
        }):
            result_with_injury = await analyst.analyze_event(
                event_id="inj_test_2",
                sport="basketball_nba",
                home="Lakers",
                away="Celtics",
                selection="Lakers",
                target_odds=2.0,
                sharp_odds=1.9,
                sharp_market={"Lakers": 1.9, "Celtics": 2.1},
            )

        # The model_p with injury should be LOWER (the fix ensures this)
        assert result_with_injury["model_probability"] <= result_no_injury["model_probability"]


class TestMomentumAdjustment:
    @pytest.mark.asyncio
    async def test_positive_momentum_increases_prob(self, analyst, mock_externals):
        result_no_momentum = await analyst.analyze_event(
            event_id="mom_1", sport="basketball_nba",
            home="A", away="B", selection="A",
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9, "B": 2.1},
            market_momentum=0.0,
        )
        result_with_momentum = await analyst.analyze_event(
            event_id="mom_2", sport="basketball_nba",
            home="A", away="B", selection="A",
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9, "B": 2.1},
            market_momentum=0.10,
        )
        assert result_with_momentum["model_probability"] > result_no_momentum["model_probability"]


class TestRecommendation:
    @pytest.mark.asyncio
    async def test_bet_when_ev_positive(self, analyst, mock_externals):
        # Force high probability for a BET recommendation
        analyst.qpm.get_true_probability.return_value = 0.75
        result = await analyst.analyze_event(
            event_id="rec_1", sport="basketball_nba",
            home="A", away="B", selection="A",
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9, "B": 2.1},
        )
        assert result["recommendation"] == "BET"

    @pytest.mark.asyncio
    async def test_skip_when_ev_negative(self, analyst, mock_externals):
        # Force low probability for a SKIP recommendation
        analyst.qpm.get_true_probability.return_value = 0.20
        result = await analyst.analyze_event(
            event_id="rec_2", sport="basketball_nba",
            home="A", away="B", selection="A",
            target_odds=2.0, sharp_odds=1.9,
            sharp_market={"A": 1.9, "B": 2.1},
        )
        assert result["recommendation"] == "SKIP"
