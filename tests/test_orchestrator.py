"""Tests for src/agents/orchestrator.py — alert dedup and pipeline coordination."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.agents.orchestrator import AgentOrchestrator


@pytest.fixture
def orchestrator():
    with patch("src.agents.orchestrator.ScoutAgent") as MockScout, \
         patch("src.agents.orchestrator.AnalystAgent") as MockAnalyst, \
         patch("src.agents.orchestrator.ExecutionerAgent") as MockExec, \
         patch("src.agents.orchestrator.PerformanceMonitor"), \
         patch("src.agents.orchestrator.cache"):
        scout = MockScout.return_value
        scout.monitor_odds = AsyncMock(return_value=[])
        scout.monitor_injuries = AsyncMock(return_value=[])

        analyst = MockAnalyst.return_value
        analyst.analyze_event = AsyncMock(return_value={
            "model_probability": 0.55, "expected_value": 0.10,
            "recommendation": "BET",
        })

        exec_agent = MockExec.return_value
        exec_agent.execute = AsyncMock(return_value={
            "action": "bet", "stake": 5.0, "reasoning": [],
        })

        orch = AgentOrchestrator()
        yield orch


class TestAlertDedup:
    @pytest.mark.asyncio
    async def test_deduplicates_same_event(self, orchestrator):
        """Multiple alerts for the same event_id should be deduped to strongest."""
        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.02},
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "B",
             "current_odds": 2.2, "prev_odds": 2.0,
             "movement_pct": 2.27, "market_momentum": -0.01},
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "Draw",
             "current_odds": 3.5, "prev_odds": 3.0,
             "movement_pct": 4.76, "market_momentum": 0.0},
        ])

        summary = await orchestrator.run_once()

        # Should only analyze ONCE (the strongest: movement_pct=5.56, selection="A")
        assert summary["analyses"] == 1
        call_args = orchestrator.analyst.analyze_event.call_args
        assert call_args.kwargs["selection"] == "A"

    @pytest.mark.asyncio
    async def test_different_events_not_deduped(self, orchestrator):
        """Alerts for different events should NOT be deduped."""
        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.0},
            {"event_id": "ev2", "type": "steam_move", "sport": "nba",
             "home": "C", "away": "D", "selection": "C",
             "current_odds": 1.7, "prev_odds": 2.0,
             "movement_pct": 8.82, "market_momentum": 0.0},
        ])

        summary = await orchestrator.run_once()
        assert summary["analyses"] == 2


class TestCommenceTimePassthrough:
    @pytest.mark.asyncio
    async def test_commence_time_attached_to_analysis(self, orchestrator):
        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.0,
             "commence_time": "2026-03-15T20:00:00Z"},
        ])

        # The analysis dict returned by the analyst gets commence_time attached
        analysis_result = {"model_probability": 0.55, "expected_value": 0.10}
        orchestrator.analyst.analyze_event = AsyncMock(return_value=analysis_result)

        await orchestrator.run_once()

        # The executioner should receive the analysis with commence_time
        exec_call = orchestrator.executioner.execute.call_args
        analysis_passed = exec_call.kwargs.get("analysis", exec_call[0][0] if exec_call[0] else {})
        assert analysis_passed.get("commence_time") == "2026-03-15T20:00:00Z"


class TestMarketMomentumPassthrough:
    @pytest.mark.asyncio
    async def test_momentum_passed_to_analyst(self, orchestrator):
        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.05},
        ])

        await orchestrator.run_once()

        call_kwargs = orchestrator.analyst.analyze_event.call_args.kwargs
        assert call_kwargs["market_momentum"] == 0.05


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_halts_on_circuit_breaker(self, orchestrator):
        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.0},
            {"event_id": "ev2", "type": "steam_move", "sport": "nba",
             "home": "C", "away": "D", "selection": "C",
             "current_odds": 1.7, "prev_odds": 2.0,
             "movement_pct": 8.82, "market_momentum": 0.0},
        ])
        # First call returns halt
        orchestrator.executioner.execute = AsyncMock(return_value={
            "action": "halt", "stake": 0.0, "reasoning": ["streak"],
        })

        summary = await orchestrator.run_once()
        assert summary["halted"] is True
        # Should stop after first halt, not process second alert
        assert summary["analyses"] == 1
