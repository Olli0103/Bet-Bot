"""Tests for src/models/compliance.py — DSS audit schemas."""
import pytest

from src.models.compliance import (
    ConfidenceBreakdown,
    HumanReviewData,
    LUGAS_MONTHLY_DEPOSIT_LIMIT,
    StatefulTip,
    TipAlert,
)


class TestHumanReviewData:
    def test_valid_review(self):
        review = HumanReviewData(
            operator_id="user123",
            confirmed_odds=2.10,
            confirmed_stake=15.0,
            action="placed",
        )
        assert review.action == "placed"
        assert review.confirmed_odds == 2.10

    def test_normalizes_action(self):
        review = HumanReviewData(
            operator_id="user123",
            confirmed_odds=2.10,
            confirmed_stake=15.0,
            action="INVALID",
        )
        assert review.action == "skipped"

    def test_all_actions_accepted(self):
        for action in ("placed", "skipped", "adjusted", "rejected"):
            review = HumanReviewData(
                operator_id="u1",
                confirmed_odds=2.0,
                confirmed_stake=10.0,
                action=action,
            )
            assert review.action == action


class TestConfidenceBreakdown:
    def test_valid_breakdown(self):
        cb = ConfidenceBreakdown(
            statistical_weight=0.5,
            market_signal_weight=0.3,
            qualitative_weight=0.2,
            top_factors=["Elo-Vorteil", "Steam Move"],
        )
        assert cb.statistical_weight == 0.5
        assert len(cb.top_factors) == 2

    def test_limits_factors_to_five(self):
        cb = ConfidenceBreakdown(
            statistical_weight=0.34,
            market_signal_weight=0.33,
            qualitative_weight=0.33,
            top_factors=["a", "b", "c", "d", "e", "f", "g"],
        )
        assert len(cb.top_factors) == 5


class TestTipAlert:
    def test_valid_alert(self):
        alert = TipAlert(
            event_id="ev1",
            match_name="Bayern vs Dortmund",
            sport="soccer",
            recommended_selection="Bayern",
            target_odds=2.10,
            signal_odds=2.15,
            model_probability=0.55,
            net_ev=0.05,
            ai_reasoning="Quantitatives Signal bestätigt durch Marktanalyse.",
        )
        assert alert.net_ev == 0.05
        assert alert.target_odds == 2.10

    def test_format_for_telegram(self):
        alert = TipAlert(
            event_id="ev1",
            match_name="Bayern vs Dortmund",
            sport="soccer",
            recommended_selection="Bayern",
            target_odds=2.10,
            signal_odds=2.15,
            model_probability=0.55,
            net_ev=0.05,
            ai_reasoning="Strong edge confirmed by model analysis.",
            confidence_breakdown=ConfidenceBreakdown(
                statistical_weight=0.5,
                market_signal_weight=0.3,
                qualitative_weight=0.2,
                top_factors=["Elo advantage"],
            ),
        )
        text = alert.format_for_telegram()
        assert "Bayern vs Dortmund" in text
        assert "Net EV" in text
        assert "Konfidenz" in text
        assert "Elo advantage" in text

    def test_rejects_invalid_odds(self):
        with pytest.raises(Exception):
            TipAlert(
                event_id="ev1",
                match_name="A vs B",
                sport="soccer",
                recommended_selection="A",
                target_odds=0.5,  # Invalid: must be > 1.0
                signal_odds=2.0,
                model_probability=0.55,
                net_ev=0.05,
                ai_reasoning="Some reasoning text here.",
            )


class TestStatefulTip:
    def test_requires_human_review(self):
        alert = TipAlert(
            event_id="ev1",
            match_name="Bayern vs Dortmund",
            sport="soccer",
            recommended_selection="Bayern",
            target_odds=2.10,
            signal_odds=2.15,
            model_probability=0.55,
            net_ev=0.05,
            ai_reasoning="Quantitatives Signal.",
        )
        tip = StatefulTip(tip_id="ev1:h2h:1234", ai_recommendation=alert)
        assert tip.requires_human_review() is True
        assert tip.is_finalized is False

    def test_finalize_with_review(self):
        alert = TipAlert(
            event_id="ev1",
            match_name="Bayern vs Dortmund",
            sport="soccer",
            recommended_selection="Bayern",
            target_odds=2.10,
            signal_odds=2.15,
            model_probability=0.55,
            net_ev=0.05,
            ai_reasoning="Quantitatives Signal.",
        )
        tip = StatefulTip(tip_id="ev1:h2h:1234", ai_recommendation=alert)

        review = HumanReviewData(
            operator_id="user123",
            confirmed_odds=2.08,
            confirmed_stake=10.0,
            action="placed",
        )
        tip.finalize(review)

        assert tip.is_finalized is True
        assert tip.requires_human_review() is False
        assert tip.human_intervention.confirmed_odds == 2.08
        assert tip.odds_at_placement == 2.08


class TestLugasLimit:
    def test_constant_value(self):
        assert LUGAS_MONTHLY_DEPOSIT_LIMIT == 1000.0
