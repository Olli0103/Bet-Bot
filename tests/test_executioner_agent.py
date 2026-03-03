"""Tests for src/agents/executioner_agent.py — caching and alert logic."""
import json
from unittest.mock import patch, MagicMock

import pytest

from src.agents.executioner_agent import ExecutionerAgent


@pytest.fixture
def executioner():
    with patch("src.agents.executioner_agent.PerformanceMonitor") as MockPM, \
         patch("src.agents.executioner_agent.BankrollManager") as MockBM:
        pm = MockPM.return_value
        pm.check_circuit_breakers.return_value = {}
        pm.get_adjustment_factors.return_value = {"min_ev": 0.01, "kelly_multiplier": 1.0}

        bm = MockBM.return_value
        bm.get_current_bankroll.return_value = 1000.0

        agent = ExecutionerAgent()
        yield agent


class TestCacheAlert:
    @patch("src.agents.executioner_agent.cache")
    def test_caches_full_analysis(self, mock_cache, executioner):
        analysis = {
            "event_id": "ev1",
            "sport": "basketball_nba",
            "home": "Lakers",
            "away": "Celtics",
            "selection": "Lakers",
            "bookmaker_odds": 2.0,
            "sharp_odds": 1.9,
            "sharp_market": {"Lakers": 1.9, "Celtics": 2.1},
            "market_momentum": 0.05,
            "trigger": "steam_move",
            "model_probability": 0.58,
            "expected_value": 0.16,
            "recommendation": "BET",
            "sentiment": {"home": 0.3, "away": -0.1},
            "injuries": {"home": 1, "away": 0},
            "injury_details": [],
            "form": {"home_wr": 0.6, "away_wr": 0.4},
            "elo": {"elo_diff": 30.0, "elo_expected": 0.6},
            "poisson_prob": None,
            "public_bias": 0.01,
            "commence_time": "2026-03-15T20:00:00Z",
        }
        alert_id = executioner._cache_alert(analysis, stake=5.0)

        # Verify cache was called
        assert mock_cache.set_json.called
        call_args = mock_cache.set_json.call_args
        cache_key = call_args[0][0]
        payload = call_args[0][1]

        assert cache_key.startswith("agent_alert:")
        assert len(alert_id) == 12

        # Verify all critical fields are cached
        assert payload["model_probability"] == 0.58
        assert payload["expected_value"] == 0.16
        assert payload["sharp_odds"] == 1.9
        assert payload["sharp_market"] == {"Lakers": 1.9, "Celtics": 2.1}
        assert payload["market_momentum"] == 0.05
        assert payload["commence_time"] == "2026-03-15T20:00:00Z"
        assert payload["sentiment"] == {"home": 0.3, "away": -0.1}
        assert payload["form"] == {"home_wr": 0.6, "away_wr": 0.4}
        assert payload["elo"] == {"elo_diff": 30.0, "elo_expected": 0.6}
        assert payload["stake"] == 5.0

    @patch("src.agents.executioner_agent.cache")
    def test_deterministic_alert_id(self, mock_cache, executioner):
        """Same analysis should produce the same alert ID."""
        analysis = {
            "event_id": "ev1", "sport": "nba", "home": "A", "away": "B",
            "selection": "A", "bookmaker_odds": 2.0, "trigger": "test",
            "model_probability": 0.55, "expected_value": 0.10,
            "recommendation": "BET",
        }
        id1 = executioner._cache_alert(analysis, stake=5.0)
        id2 = executioner._cache_alert(analysis, stake=5.0)
        assert id1 == id2


class TestExecute:
    @pytest.mark.asyncio
    @patch("src.agents.executioner_agent.get_reliability_adjustment", return_value=1.0)
    async def test_skip_low_ev(self, mock_adj, executioner):
        analysis = {
            "event_id": "ev1", "sport": "basketball_nba",
            "selection": "Lakers", "model_probability": 0.30,
            "expected_value": -0.10, "recommendation": "SKIP",
            "bookmaker_odds": 2.0, "trigger": "test",
        }
        result = await executioner.execute(analysis)
        assert result["action"] == "skip"
        assert result["stake"] == 0.0

    @pytest.mark.asyncio
    @patch("src.agents.executioner_agent.get_reliability_adjustment", return_value=1.0)
    async def test_bet_positive_ev(self, mock_adj, executioner):
        analysis = {
            "event_id": "ev1", "sport": "basketball_nba",
            "selection": "Lakers", "model_probability": 0.65,
            "expected_value": 0.30, "recommendation": "BET",
            "bookmaker_odds": 2.0, "trigger": "test",
            "features": {},
        }
        # Need to mock ghost trading
        with patch("src.agents.executioner_agent.place_virtual_bet", create=True):
            with patch.dict("sys.modules", {
                "src.core.ghost_trading": MagicMock(place_virtual_bet=MagicMock()),
            }):
                result = await executioner.execute(analysis)
        assert result["action"] == "bet"
        assert result["stake"] > 0

    @pytest.mark.asyncio
    async def test_halt_on_losing_streak(self, executioner):
        executioner.monitor.check_circuit_breakers.return_value = {"losing_streak": True}
        analysis = {
            "event_id": "ev1", "model_probability": 0.65,
            "expected_value": 0.30, "recommendation": "BET",
        }
        result = await executioner.execute(analysis)
        assert result["action"] == "halt"
