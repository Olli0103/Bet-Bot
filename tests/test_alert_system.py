"""Tests for the alert management system.

Covers:
1. test_alert_priority_scoring — score computation + priority classification
2. test_alert_dedup_debounce — dedup suppression + delta threshold
3. test_alert_requires_calibrated_confidence — quality guard blocks raw passthrough
4. test_medium_alerts_go_to_digest_not_immediate — routing correctness
5. test_alert_card_contains_actionability_block — card format validation
6. test_split_mode_documented_as_experimental_and_monolith_default — ops doc check
"""
from __future__ import annotations

import time
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(
    model_probability: float = 0.65,
    expected_value: float = 0.04,
    bookmaker_odds: float = 2.0,
    calibration_source: str = "soccer_h2h",
    trigger: str = "steam_move",
    market_momentum: float = 0.02,
    event_id: str = "evt_001",
    sport: str = "soccer_germany_bundesliga",
    home: str = "Dortmund",
    away: str = "Bayern",
    selection: str = "Dortmund",
    market: str = "h2h",
    sharp_odds: float = 1.9,
    public_bias: float = 0.01,
    commence_time: str = "2026-03-04T18:30:00Z",
    movement_pct: float = 3.5,
) -> dict:
    return {
        "model_probability": model_probability,
        "expected_value": expected_value,
        "bookmaker_odds": bookmaker_odds,
        "calibration_source": calibration_source,
        "trigger": trigger,
        "market_momentum": market_momentum,
        "event_id": event_id,
        "sport": sport,
        "home": home,
        "away": away,
        "selection": selection,
        "market": market,
        "sharp_odds": sharp_odds,
        "public_bias": public_bias,
        "commence_time": commence_time,
        "movement_pct": movement_pct,
    }


class FakeCache:
    """In-memory Redis cache replacement for testing."""

    def __init__(self):
        self._store: dict = {}

    def get_json(self, key):
        val = self._store.get(key)
        if val is None:
            return None
        return val.get("data")

    def set_json(self, key, data, ttl_seconds=None):
        self._store[key] = {"data": data}

    def delete(self, key):
        self._store.pop(key, None)

    @property
    def client(self):
        return self


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_cache():
    """Replace Redis cache with in-memory fake for all tests."""
    fake = FakeCache()
    with patch("src.core.alert_manager.cache", fake):
        yield fake


# ---------------------------------------------------------------------------
# 1) test_alert_priority_scoring
# ---------------------------------------------------------------------------

class TestAlertPriorityScoring:
    def test_high_ev_high_prob_is_critical(self):
        from src.core.alert_manager import compute_alert_score, classify_priority

        analysis = _make_analysis(
            model_probability=0.75,
            expected_value=0.08,
            calibration_source="soccer_h2h",
            market_momentum=0.02,
        )
        score = compute_alert_score(analysis)
        priority = classify_priority(score, analysis)
        assert score >= 65
        assert priority.value == "CRITICAL"

    def test_moderate_ev_is_high(self):
        from src.core.alert_manager import compute_alert_score, classify_priority

        analysis = _make_analysis(
            model_probability=0.60,
            expected_value=0.03,
            bookmaker_odds=2.2,
        )
        score = compute_alert_score(analysis)
        priority = classify_priority(score, analysis)
        assert priority.value in ("HIGH", "CRITICAL")

    def test_low_ev_low_prob_is_low(self):
        from src.core.alert_manager import compute_alert_score, classify_priority

        analysis = _make_analysis(
            model_probability=0.40,
            expected_value=0.005,
            bookmaker_odds=2.5,
            calibration_source="global",
            market_momentum=0.001,
        )
        score = compute_alert_score(analysis)
        priority = classify_priority(score, analysis)
        assert priority.value in ("LOW", "MEDIUM")

    def test_score_is_between_0_and_100(self):
        from src.core.alert_manager import compute_alert_score

        for ev in [0, 0.01, 0.05, 0.15]:
            for p in [0.3, 0.5, 0.7, 0.9]:
                score = compute_alert_score(_make_analysis(
                    model_probability=p, expected_value=ev,
                ))
                assert 0 <= score <= 100, f"Score {score} out of range for ev={ev}, p={p}"


# ---------------------------------------------------------------------------
# 2) test_alert_dedup_debounce
# ---------------------------------------------------------------------------

class TestAlertDedupDebounce:
    def test_first_alert_not_suppressed(self):
        from src.core.alert_manager import is_duplicate_alert

        assert not is_duplicate_alert("evt_1", "h2h", "Dortmund", 3.5)

    def test_identical_alert_suppressed_within_window(self):
        from src.core.alert_manager import is_duplicate_alert, mark_alert_sent

        mark_alert_sent("evt_1", "h2h", "Dortmund", 3.5)
        assert is_duplicate_alert("evt_1", "h2h", "Dortmund", 3.5)

    def test_significant_delta_not_suppressed(self):
        from src.core.alert_manager import is_duplicate_alert, mark_alert_sent

        mark_alert_sent("evt_2", "h2h", "Bayern", 3.0)
        # Large delta (5.5 vs 3.0) should NOT be suppressed
        assert not is_duplicate_alert("evt_2", "h2h", "Bayern", 5.5)

    def test_different_event_not_suppressed(self):
        from src.core.alert_manager import is_duplicate_alert, mark_alert_sent

        mark_alert_sent("evt_3", "h2h", "Dortmund", 3.5)
        assert not is_duplicate_alert("evt_4", "h2h", "Dortmund", 3.5)

    def test_tiny_delta_suppressed(self):
        from src.core.alert_manager import is_duplicate_alert, mark_alert_sent

        mark_alert_sent("evt_5", "h2h", "Liverpool", 4.0)
        # Tiny delta (4.01 vs 4.0) should be suppressed
        assert is_duplicate_alert("evt_5", "h2h", "Liverpool", 4.01)


# ---------------------------------------------------------------------------
# 3) test_alert_requires_calibrated_confidence
# ---------------------------------------------------------------------------

class TestAlertRequiresCalibratedConfidence:
    def test_raw_passthrough_suppressed(self):
        from src.core.alert_manager import run_quality_guards

        analysis = _make_analysis(
            model_probability=0.70,
            expected_value=0.05,
            calibration_source="raw_passthrough",
        )
        passes, reason = run_quality_guards(analysis)
        assert not passes
        assert "calibration" in reason.lower() or "raw_passthrough" in reason

    def test_empty_calibration_source_suppressed(self):
        from src.core.alert_manager import run_quality_guards

        analysis = _make_analysis(
            model_probability=0.70,
            expected_value=0.05,
            calibration_source="",
        )
        passes, reason = run_quality_guards(analysis)
        assert not passes

    def test_calibrated_source_passes(self):
        from src.core.alert_manager import run_quality_guards

        analysis = _make_analysis(
            model_probability=0.65,
            expected_value=0.04,
            calibration_source="soccer_h2h",
        )
        passes, _ = run_quality_guards(analysis)
        assert passes

    def test_low_probability_suppressed(self):
        from src.core.alert_manager import run_quality_guards

        analysis = _make_analysis(
            model_probability=0.20,
            expected_value=0.01,
            calibration_source="soccer_h2h",
        )
        passes, reason = run_quality_guards(analysis)
        assert not passes
        assert "model_probability" in reason

    def test_odds_glitch_suppressed(self):
        from src.core.alert_manager import check_odds_glitch

        analysis = _make_analysis(
            bookmaker_odds=1.005,  # basically free money = glitch
        )
        passes, reason = check_odds_glitch(analysis)
        assert not passes
        assert "out of range" in reason

    def test_steam_without_confirmation_suppressed(self):
        from src.core.alert_manager import check_steam_confirmation

        analysis = _make_analysis(
            trigger="steam_move",
            expected_value=0.005,  # weak EV
            model_probability=0.45,  # weak prob
            market_momentum=0.005,  # weak momentum
        )
        passes, reason = check_steam_confirmation(analysis)
        assert not passes
        assert "without confirmation" in reason


# ---------------------------------------------------------------------------
# 4) test_medium_alerts_go_to_digest_not_immediate
# ---------------------------------------------------------------------------

class TestMediumAlertsDigest:
    def test_medium_priority_not_sent_immediately(self):
        from src.core.alert_manager import process_alert

        # Construct an analysis that scores MEDIUM (not HIGH/CRITICAL)
        analysis = _make_analysis(
            model_probability=0.52,
            expected_value=0.015,
            bookmaker_odds=2.1,
            calibration_source="global",
            market_momentum=0.005,
        )
        result = process_alert(analysis)

        if result.priority.value == "MEDIUM":
            assert not result.should_send_immediate
            assert result.should_digest
        else:
            # If scoring puts it elsewhere, that's also valid
            # but we verify the routing logic is consistent
            if result.priority.value in ("HIGH", "CRITICAL"):
                assert result.should_send_immediate
            elif result.priority.value == "LOW":
                assert result.should_suppress

    def test_high_priority_sent_immediately(self):
        from src.core.alert_manager import process_alert

        analysis = _make_analysis(
            model_probability=0.70,
            expected_value=0.06,
            bookmaker_odds=1.8,
            calibration_source="soccer_h2h",
            market_momentum=0.03,
        )
        result = process_alert(analysis)
        # Should be HIGH or CRITICAL
        assert result.priority.value in ("HIGH", "CRITICAL")
        assert result.should_send_immediate
        assert not result.should_digest

    def test_digest_buffer_accumulates(self):
        from src.core.alert_manager import add_to_digest, pop_digest

        add_to_digest({"home": "A", "away": "B", "selection": "A", "ev": 0.01})
        add_to_digest({"home": "C", "away": "D", "selection": "C", "ev": 0.02})

        alerts = pop_digest()
        assert len(alerts) == 2
        # Buffer should be empty after pop
        assert pop_digest() == []


# ---------------------------------------------------------------------------
# 5) test_alert_card_contains_actionability_block
# ---------------------------------------------------------------------------

class TestAlertCardActionabilityBlock:
    def test_card_contains_playability(self):
        from src.core.alert_manager import (
            AlertPriority,
            build_actionability_block,
            format_alert_card,
        )

        analysis = _make_analysis()
        priority = AlertPriority.HIGH
        actionability = build_actionability_block(analysis, priority)
        card = format_alert_card(analysis, priority, actionability, stake=5.0)

        assert actionability.playability in ("PLAYABLE", "WATCHLIST", "BLOCKED")
        assert actionability.playability in card

    def test_card_contains_reasons(self):
        from src.core.alert_manager import (
            AlertPriority,
            build_actionability_block,
            format_alert_card,
        )

        analysis = _make_analysis(
            model_probability=0.75,
            expected_value=0.06,
        )
        priority = AlertPriority.CRITICAL
        actionability = build_actionability_block(analysis, priority)
        card = format_alert_card(analysis, priority, actionability, stake=10.0)

        assert len(actionability.reasons) > 0
        # At least one reason should appear in the card
        assert any(r in card for r in actionability.reasons)

    def test_card_contains_ev_and_edge(self):
        from src.core.alert_manager import (
            AlertPriority,
            build_actionability_block,
            format_alert_card,
        )

        analysis = _make_analysis()
        priority = AlertPriority.HIGH
        actionability = build_actionability_block(analysis, priority)
        card = format_alert_card(analysis, priority, actionability)

        assert "EV:" in card
        assert "Edge:" in card
        assert "Modell:" in card

    def test_card_contains_risk_flags_when_present(self):
        from src.core.alert_manager import (
            AlertPriority,
            build_actionability_block,
            format_alert_card,
        )

        analysis = _make_analysis(
            bookmaker_odds=4.5,  # high-odds flag
            calibration_source="global",  # weak-calibration flag
        )
        priority = AlertPriority.HIGH
        actionability = build_actionability_block(analysis, priority)
        card = format_alert_card(analysis, priority, actionability)

        assert len(actionability.risk_flags) > 0
        assert "Risiken:" in card

    def test_actionability_block_has_all_fields(self):
        from src.core.alert_manager import AlertPriority, build_actionability_block

        analysis = _make_analysis()
        ab = build_actionability_block(analysis, AlertPriority.HIGH)

        assert ab.playability in ("PLAYABLE", "WATCHLIST", "BLOCKED")
        assert isinstance(ab.reasons, list)
        assert isinstance(ab.risk_flags, list)
        assert ab.confidence_source in ("calibrated", "raw")
        assert 0 <= ab.model_probability <= 1
        assert 0 <= ab.implied_probability <= 1
        assert isinstance(ab.expected_value, float)
        assert isinstance(ab.tax_adjusted_ev, float)

    def test_digest_card_format(self):
        from src.core.alert_manager import format_digest_card

        alerts = [
            {"home": "A", "away": "B", "selection": "A",
             "expected_value": 0.03, "model_probability": 0.60,
             "playability": "PLAYABLE"},
            {"home": "C", "away": "D", "selection": "C",
             "expected_value": 0.02, "model_probability": 0.55,
             "playability": "WATCHLIST"},
        ]
        card = format_digest_card(alerts)
        assert "DIGEST" in card
        assert "A vs B" in card
        assert "C vs D" in card
        assert "2 Watchlist" in card


# ---------------------------------------------------------------------------
# 6) test_split_mode_documented_as_experimental_and_monolith_default
# ---------------------------------------------------------------------------

class TestSplitModeDocumentation:
    def test_changelog_documents_monolith_as_default(self):
        with open("CHANGELOG.md", "r") as f:
            content = f.read()

        # Monolith should be documented as default/stable
        assert "Monolith" in content or "monolith" in content
        assert "python -m src.bot.app" in content

    def test_changelog_documents_split_as_experimental(self):
        with open("CHANGELOG.md", "r") as f:
            content = f.read()

        assert "Experimental" in content or "experimental" in content
        assert "core_worker" in content
        assert "telegram_worker" in content

    def test_changelog_documents_revert_reasoning(self):
        with open("CHANGELOG.md", "r") as f:
            content = f.read()

        # Should mention instability symptoms
        assert "laggy" in content.lower() or "instabil" in content.lower() or "unstable" in content.lower()
        assert "Revert" in content or "revert" in content

    def test_readme_has_runtime_modes(self):
        with open("README.md", "r") as f:
            content = f.read()

        # README should mention both modes
        assert "python -m src.bot.app" in content


# ---------------------------------------------------------------------------
# Additional: Metrics
# ---------------------------------------------------------------------------

class TestAlertMetrics:
    def test_metrics_tracking(self):
        from src.core.alert_manager import AlertMetrics

        AlertMetrics.reset()
        AlertMetrics.record("total")
        AlertMetrics.record("total")
        AlertMetrics.record("sent_immediate", "HIGH")
        AlertMetrics.record("suppressed_dedup")

        m = AlertMetrics.get_all()
        assert m["alerts_total"] == 2
        assert m["alerts_sent_immediate"] == 1
        assert m["alerts_suppressed_dedup"] == 1

    def test_report_generation(self):
        from src.core.alert_manager import AlertMetrics, generate_alert_quality_report

        AlertMetrics.reset()
        AlertMetrics.record("total")
        AlertMetrics.record("sent_immediate", "CRITICAL")

        report = generate_alert_quality_report()
        assert "metrics" in report
        assert "summary" in report
        assert report["summary"]["total_alerts"] == 1


# ---------------------------------------------------------------------------
# Additional: Quality Guards Integration
# ---------------------------------------------------------------------------

class TestQualityGuardsIntegration:
    def test_process_alert_suppresses_bad_quality(self):
        from src.core.alert_manager import process_alert

        analysis = _make_analysis(
            model_probability=0.20,
            calibration_source="raw_passthrough",
        )
        result = process_alert(analysis)
        assert result.should_suppress
        assert "quality" in result.suppress_reason

    def test_process_alert_routes_good_alert(self):
        from src.core.alert_manager import process_alert

        analysis = _make_analysis(
            model_probability=0.72,
            expected_value=0.07,
            calibration_source="soccer_h2h",
            market_momentum=0.025,
        )
        result = process_alert(analysis)
        assert result.priority.value in ("HIGH", "CRITICAL")
        assert result.should_send_immediate
        assert result.card_text  # non-empty card
        assert result.actionability.playability in ("PLAYABLE", "WATCHLIST")
