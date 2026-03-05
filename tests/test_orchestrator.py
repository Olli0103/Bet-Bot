"""Tests for src/agents/orchestrator.py — alert dedup, DSS tip flow, and DLQ."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from src.agents.orchestrator import AgentOrchestrator
from src.agents.tip_publisher import TipState, TipStatus


@pytest.fixture
def orchestrator():
    with patch("src.agents.orchestrator.ScoutAgent") as MockScout, \
         patch("src.agents.orchestrator.AnalystAgent") as MockAnalyst, \
         patch("src.agents.orchestrator.PerformanceMonitor") as MockPM, \
         patch("src.agents.orchestrator.cache"):
        scout = MockScout.return_value
        scout.monitor_odds = AsyncMock(return_value=[])
        scout.monitor_injuries = AsyncMock(return_value=[])

        analyst = MockAnalyst.return_value
        analyst.analyze_event = AsyncMock(return_value={
            "model_probability": 0.55, "expected_value": 0.10,
            "recommendation": "BET",
        })

        pm = MockPM.return_value
        pm.check_circuit_breakers.return_value = {}

        orch = AgentOrchestrator()
        # Mock run_tip_flow to return a published TipState
        published_state = TipState(
            event_id="ev1", sport="nba", home="A", away="B",
            selection="A", market="h2h", initial_odds=2.0,
        )
        published_state.status = TipStatus.PUBLISHED
        orch.run_tip_flow = AsyncMock(return_value=published_state)
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

        # Should only process ONCE (the strongest: movement_pct=5.56, selection="A")
        assert summary["analyses"] == 1
        assert orchestrator.run_tip_flow.call_count == 1

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
    async def test_commence_time_in_alert(self, orchestrator):
        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.0,
             "commence_time": "2026-03-15T20:00:00Z"},
        ])

        await orchestrator.run_once()

        # The run_tip_flow should receive the alert with commence_time
        call_args = orchestrator.run_tip_flow.call_args
        alert_passed = call_args[0][0]
        assert alert_passed.get("commence_time") == "2026-03-15T20:00:00Z"


class TestMarketMomentumPassthrough:
    @pytest.mark.asyncio
    async def test_momentum_in_alert(self, orchestrator):
        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.05},
        ])

        await orchestrator.run_once()

        call_args = orchestrator.run_tip_flow.call_args
        alert_passed = call_args[0][0]
        assert alert_passed.get("market_momentum") == 0.05


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
        # Circuit breaker tripped
        orchestrator.monitor.check_circuit_breakers.return_value = {"losing_streak": True}

        summary = await orchestrator.run_once()
        assert summary["halted"] is True
        # Should not process any alerts when halted
        assert summary["analyses"] == 0


class TestDSSFlow:
    @pytest.mark.asyncio
    async def test_published_tip_counted(self, orchestrator):
        """Published tips should be counted in summary."""
        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.0},
        ])

        summary = await orchestrator.run_once()
        assert summary["tips_published"] == 1
        assert summary["tips_rejected"] == 0

    @pytest.mark.asyncio
    async def test_rejected_tip_counted(self, orchestrator):
        """Rejected tips should be counted in summary."""
        rejected_state = TipState(
            event_id="ev1", sport="nba", home="A", away="B",
            selection="A", market="h2h", initial_odds=2.0,
        )
        rejected_state.status = TipStatus.REJECTED
        rejected_state.rejection_reason = "Negative EV"
        orchestrator.run_tip_flow = AsyncMock(return_value=rejected_state)

        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.0},
        ])

        summary = await orchestrator.run_once()
        assert summary["tips_published"] == 0
        assert summary["tips_rejected"] == 1


class TestDeadLetterQueue:
    @pytest.mark.asyncio
    async def test_failed_alerts_enqueued_to_dlq(self, orchestrator):
        """Transient failures should be added to the DLQ."""
        orchestrator.run_tip_flow = AsyncMock(side_effect=Exception("API timeout"))

        orchestrator.scout.monitor_odds = AsyncMock(return_value=[
            {"event_id": "ev1", "type": "steam_move", "sport": "nba",
             "home": "A", "away": "B", "selection": "A",
             "current_odds": 1.8, "prev_odds": 2.0,
             "movement_pct": 5.56, "market_momentum": 0.0},
        ])

        summary = await orchestrator.run_once()
        assert len(orchestrator._dead_letter_queue) == 1
        dlq_alert = orchestrator._dead_letter_queue[0]
        assert dlq_alert["_dlq_retries"] == 1
        assert "API timeout" in dlq_alert["_dlq_error"]

    def test_dlq_max_retries_drops_alert(self, orchestrator):
        """Alerts exceeding max retries should be permanently dropped."""
        alert = {"event_id": "ev1", "market": "h2h", "_dlq_retries": 3}
        orchestrator._enqueue_dead_letter(alert, "still failing")
        assert len(orchestrator._dead_letter_queue) == 0

    @pytest.mark.asyncio
    async def test_dlq_retry_on_quiet_period(self, orchestrator):
        """DLQ alerts should be retried when no new alerts arrive."""
        # Pre-populate DLQ
        orchestrator._dead_letter_queue.append({
            "event_id": "ev1", "sport": "nba", "home": "A", "away": "B",
            "selection": "A", "market": "h2h", "current_odds": 1.8,
            "_dlq_retries": 1,
        })

        # Reset run_tip_flow to succeed
        published_state = TipState(
            event_id="ev1", sport="nba", home="A", away="B",
            selection="A", market="h2h", initial_odds=1.8,
        )
        published_state.status = TipStatus.PUBLISHED
        orchestrator.run_tip_flow = AsyncMock(return_value=published_state)

        # No new alerts — should trigger DLQ retry
        summary = await orchestrator.run_once()
        assert len(orchestrator._dead_letter_queue) == 0
