"""Tests for src/agents/tip_publisher.py — stateful tip flow."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.tip_publisher import (
    MAX_REVALIDATION_ATTEMPTS,
    ODDS_DRIFT_THRESHOLD,
    TipState,
    TipStatus,
    has_odds_drifted,
    tip_flow,
    validate_tip,
)


@pytest.fixture
def base_state():
    return TipState(
        event_id="ev1",
        sport="soccer_germany_bundesliga",
        home="Bayern",
        away="Dortmund",
        selection="Bayern",
        market="h2h",
        initial_odds=2.0,
    )


class TestTipState:
    def test_initial_status(self, base_state):
        assert base_state.status == TipStatus.DISCOVERED

    def test_to_dict(self, base_state):
        d = base_state.to_dict()
        assert d["event_id"] == "ev1"
        assert d["status"] == "DISCOVERED"
        assert d["initial_odds"] == 2.0
        assert d["revalidation_count"] == 0


class TestHasOddsDrifted:
    def test_no_drift(self, base_state):
        base_state.current_odds = 2.0
        assert not has_odds_drifted(base_state)

    def test_small_drift_below_threshold(self, base_state):
        # 2.0 → 2.05: implied prob shift = 0.5 - 0.4878 = 0.0122 < 0.03
        base_state.current_odds = 2.05
        assert not has_odds_drifted(base_state)

    def test_large_drift_above_threshold(self, base_state):
        # 2.0 → 2.20: implied prob shift = 0.5 - 0.4545 = 0.0455 > 0.03
        base_state.current_odds = 2.20
        assert has_odds_drifted(base_state)

    def test_degenerate_odds(self, base_state):
        base_state.current_odds = 1.0
        assert has_odds_drifted(base_state)


class TestValidateTip:
    def test_valid_tip(self, base_state):
        base_state.model_probability = 0.60
        base_state.current_odds = 2.0
        assert validate_tip(base_state) is True
        assert base_state.expected_value > 0

    def test_negative_ev_rejected(self, base_state):
        base_state.model_probability = 0.40
        base_state.current_odds = 2.0
        assert validate_tip(base_state) is False
        assert base_state.status == TipStatus.REJECTED
        assert "Negative tax-adjusted EV" in base_state.rejection_reason

    def test_below_mao_rejected(self, base_state):
        # High model prob but very low odds (below MAO)
        base_state.model_probability = 0.90
        base_state.current_odds = 1.05  # very low odds
        assert validate_tip(base_state) is False
        assert base_state.status == TipStatus.REJECTED


class TestTipFlow:
    @pytest.mark.asyncio
    async def test_successful_publish(self, base_state):
        analyst = MagicMock()
        analyst.analyze_event = AsyncMock(return_value={
            "model_probability": 0.60,
            "expected_value": 0.10,
        })
        get_odds = AsyncMock(return_value=2.0)
        publish = AsyncMock()

        result = await tip_flow(base_state, analyst, get_odds, publish)

        assert result.status == TipStatus.PUBLISHED
        assert result.model_probability == 0.60
        assert publish.await_count == 1

    @pytest.mark.asyncio
    async def test_rejected_on_negative_ev(self, base_state):
        analyst = MagicMock()
        analyst.analyze_event = AsyncMock(return_value={
            "model_probability": 0.40,
        })
        get_odds = AsyncMock(return_value=2.0)
        publish = AsyncMock()

        result = await tip_flow(base_state, analyst, get_odds, publish)

        assert result.status == TipStatus.REJECTED
        assert publish.await_count == 0

    @pytest.mark.asyncio
    async def test_revalidation_on_drift(self, base_state):
        analyst = MagicMock()
        analyst.analyze_event = AsyncMock(return_value={
            "model_probability": 0.60,
        })
        # First call returns drifted odds, second returns stable
        get_odds = AsyncMock(side_effect=[2.30, 2.30])
        publish = AsyncMock()

        result = await tip_flow(base_state, analyst, get_odds, publish)

        # Should have re-validated at least once due to drift
        assert result.revalidation_count >= 1

    @pytest.mark.asyncio
    async def test_analysis_failure_rejects(self, base_state):
        analyst = MagicMock()
        analyst.analyze_event = AsyncMock(side_effect=RuntimeError("API down"))
        get_odds = AsyncMock(return_value=2.0)
        publish = AsyncMock()

        result = await tip_flow(base_state, analyst, get_odds, publish)

        assert result.status == TipStatus.REJECTED
        assert "Analysis failed" in result.rejection_reason
        assert publish.await_count == 0
