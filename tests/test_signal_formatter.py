"""Tests for signal card formatting, deduplication, status badges,
sorting, and summary counts."""
import pytest


# ── 1) test_ui_no_information_loss_signal_card ──

class TestUINoInformationLoss:
    """Verify new card format contains ALL fields from the old format."""

    def test_card_contains_all_required_fields(self):
        from src.utils.signal_formatter import format_signal_card

        signal = {
            "sport": "soccer_epl",
            "market": "h2h",
            "selection": "Arsenal (Arsenal vs Chelsea)",
            "bookmaker_odds": 2.10,
            "model_probability": 0.62,
            "expected_value": 0.05,
            "recommended_stake": 15.0,
            "source_mode": "primary",
            "reference_book": "pinnacle",
            "source_quality": 1.0,
            "kelly_raw": 0.035,
            "kelly_fraction": 0.035,
            "stake_before_cap": 18.0,
            "stake_cap_applied": True,
            "trigger": "steam_move",
            "rejected_reason": "",
            "explanation": "Strong home form + value edge",
            "confidence": 0.62,
        }

        card = format_signal_card(signal, 0, 5)

        # All critical fields must appear
        assert "Arsenal" in card
        assert "2.10" in card
        assert "62%" in card              # model_probability
        assert "+0.0500" in card          # EV
        assert "100%" in card             # source_quality
        assert "0.0350" in card           # kelly_raw
        assert "18.00" in card            # stake_before_cap
        assert "15.00" in card            # recommended_stake
        assert "[CAP]" in card            # cap applied
        assert "steam_move" in card       # trigger
        assert "primary" in card          # source_mode
        assert "pinnacle" in card         # reference_book
        assert "Strong home form" in card # explanation

    def test_card_has_status_badge(self):
        from src.utils.signal_formatter import format_signal_card

        signal = {
            "sport": "soccer_epl",
            "market": "h2h",
            "selection": "Test",
            "bookmaker_odds": 2.0,
            "model_probability": 0.60,
            "expected_value": 0.05,
            "recommended_stake": 10.0,
            "rejected_reason": "",
            "display_status": "PLAYABLE",
            "display_badge": "\U0001F7E2",
        }
        card = format_signal_card(signal, 0, 1)
        assert "PLAYABLE" in card
        assert "\U0001F7E2" in card


# ── 2) test_one_pick_per_event_market_group_in_ui ──

class TestOnePickPerEventMarketGroup:
    """Verify deduplication keeps one pick per (event_id, canonical_market_group)."""

    def test_dedup_keeps_best_per_group(self):
        from src.utils.signal_formatter import deduplicate_signals

        signals = [
            {
                "event_id": "evt1",
                "market": "h2h",
                "selection": "Team A",
                "model_probability": 0.60,
                "expected_value": 0.05,
                "bookmaker_odds": 2.0,
                "recommended_stake": 10.0,
                "rejected_reason": "",
            },
            {
                "event_id": "evt1",
                "market": "h2h",
                "selection": "Team B",
                "model_probability": 0.55,
                "expected_value": 0.03,
                "bookmaker_odds": 2.5,
                "recommended_stake": 8.0,
                "rejected_reason": "",
            },
            {
                "event_id": "evt1",
                "market": "spreads +1.5",
                "selection": "Team A +1.5",
                "model_probability": 0.70,
                "expected_value": 0.08,
                "bookmaker_odds": 1.8,
                "recommended_stake": 12.0,
                "rejected_reason": "",
            },
        ]
        result = deduplicate_signals(signals)

        # Should have 2: one h2h, one spreads
        assert len(result) == 2

        groups = {s["canonical_market_group"] for s in result}
        assert "h2h" in groups
        assert "spreads" in groups

        # H2H winner should be Team A (higher model_probability)
        h2h_pick = [s for s in result if s["canonical_market_group"] == "h2h"][0]
        assert h2h_pick["selection"] == "Team A"
        assert h2h_pick["model_probability"] == 0.60

    def test_dedup_different_events_kept(self):
        from src.utils.signal_formatter import deduplicate_signals

        signals = [
            {
                "event_id": "evt1",
                "market": "h2h",
                "selection": "Team A",
                "model_probability": 0.60,
                "expected_value": 0.05,
                "bookmaker_odds": 2.0,
                "recommended_stake": 10.0,
                "rejected_reason": "",
            },
            {
                "event_id": "evt2",
                "market": "h2h",
                "selection": "Team C",
                "model_probability": 0.58,
                "expected_value": 0.04,
                "bookmaker_odds": 2.2,
                "recommended_stake": 9.0,
                "rejected_reason": "",
            },
        ]
        result = deduplicate_signals(signals)
        assert len(result) == 2  # Different events, same market -> both kept

    def test_canonical_market_group_mapping(self):
        from src.utils.signal_formatter import canonical_market_group

        assert canonical_market_group("h2h") == "h2h"
        assert canonical_market_group("moneyline") == "h2h"
        assert canonical_market_group("double_chance 1X") == "double_chance"
        assert canonical_market_group("draw_no_bet") == "draw_no_bet"
        assert canonical_market_group("spreads +1.5") == "spreads"
        assert canonical_market_group("totals 2.5") == "totals"
        assert canonical_market_group("over_under") == "totals"
        assert canonical_market_group("unknown_market") == "h2h"  # fallback


# ── 3) test_ui_status_badges_playable_watchlist_blocked ──

class TestUIStatusBadges:
    """Verify correct status badge assignment."""

    def test_playable_badge(self):
        from src.utils.signal_formatter import display_status

        sig = {
            "recommended_stake": 10.0,
            "expected_value": 0.05,
            "rejected_reason": "",
        }
        badge, label, reason = display_status(sig)
        assert label == "PLAYABLE"
        assert badge == "\U0001F7E2"
        assert reason == ""

    def test_watchlist_badge_negative_ev(self):
        from src.utils.signal_formatter import display_status

        sig = {
            "recommended_stake": 10.0,
            "expected_value": -0.01,
            "rejected_reason": "",
        }
        badge, label, reason = display_status(sig)
        assert label == "WATCHLIST"
        assert badge == "\U0001F7E1"

    def test_blocked_badge_confidence(self):
        from src.utils.signal_formatter import display_status

        sig = {
            "recommended_stake": 0.0,
            "expected_value": 0.05,
            "rejected_reason": "reject_confidence_below_min: model_prob=0.45 < gate=0.55",
        }
        badge, label, reason = display_status(sig)
        assert label == "BLOCKED"
        assert badge == "\U0001F534"
        assert "confidence" in reason.lower()

    def test_blocked_badge_zero_stake(self):
        from src.utils.signal_formatter import display_status

        sig = {
            "recommended_stake": 0.0,
            "expected_value": 0.05,
            "rejected_reason": "",
        }
        badge, label, reason = display_status(sig)
        assert label == "BLOCKED"


# ── 4) test_ui_sorting_confidence_then_ev ──

class TestUISorting:
    """Verify signals are sorted by model_probability DESC, EV DESC, odds ASC."""

    def test_sort_order(self):
        from src.utils.signal_formatter import sort_signals

        signals = [
            {"model_probability": 0.55, "expected_value": 0.03, "bookmaker_odds": 2.0},
            {"model_probability": 0.65, "expected_value": 0.05, "bookmaker_odds": 1.8},
            {"model_probability": 0.65, "expected_value": 0.05, "bookmaker_odds": 2.2},
            {"model_probability": 0.65, "expected_value": 0.02, "bookmaker_odds": 1.5},
            {"model_probability": 0.70, "expected_value": 0.01, "bookmaker_odds": 3.0},
        ]
        sorted_sigs = sort_signals(signals)

        # First: highest model_probability
        assert sorted_sigs[0]["model_probability"] == 0.70

        # Next two: same model_probability=0.65, sorted by EV DESC
        assert sorted_sigs[1]["model_probability"] == 0.65
        assert sorted_sigs[1]["expected_value"] == 0.05
        # Among same model_prob + same EV, lower odds first (ASC = -odds in key)
        assert sorted_sigs[1]["bookmaker_odds"] == 1.8
        assert sorted_sigs[2]["bookmaker_odds"] == 2.2

        # Then lower EV
        assert sorted_sigs[3]["expected_value"] == 0.02

        # Last: lowest model_probability
        assert sorted_sigs[4]["model_probability"] == 0.55


# ── 5) test_ui_summary_counts_match_payload ──

class TestUISummaryCounts:
    """Verify summary header counts match actual payload."""

    def test_summary_format(self):
        from src.utils.signal_formatter import format_summary_header

        summary = format_summary_header(
            raw_count=42,
            deduped_count=15,
            statuses={"PLAYABLE": 10, "WATCHLIST": 3, "BLOCKED": 2},
            ev_cut=0.01,
            conf_gates="Soccer=0.55 Tennis=0.57",
        )

        assert "42" in summary
        assert "15" in summary
        assert "PLAYABLE: 10" in summary
        assert "WATCHLIST: 3" in summary
        assert "BLOCKED: 2" in summary
        assert "0.010" in summary
        assert "Soccer=0.55" in summary

    def test_summary_with_dedup_signals(self):
        from src.utils.signal_formatter import (
            deduplicate_signals,
            display_status,
            format_summary_header,
        )

        signals = [
            {
                "event_id": "evt1", "market": "h2h", "selection": "A",
                "model_probability": 0.65, "expected_value": 0.05,
                "bookmaker_odds": 2.0, "recommended_stake": 10.0,
                "rejected_reason": "",
            },
            {
                "event_id": "evt1", "market": "h2h", "selection": "B",
                "model_probability": 0.55, "expected_value": 0.02,
                "bookmaker_odds": 2.5, "recommended_stake": 5.0,
                "rejected_reason": "",
            },
            {
                "event_id": "evt2", "market": "h2h", "selection": "C",
                "model_probability": 0.50, "expected_value": -0.01,
                "bookmaker_odds": 3.0, "recommended_stake": 3.0,
                "rejected_reason": "",
            },
        ]

        deduped = deduplicate_signals(signals)
        assert len(deduped) == 2  # evt1 h2h deduped, evt2 h2h kept

        statuses = {}
        for s in deduped:
            _, label, _ = display_status(s)
            statuses[label] = statuses.get(label, 0) + 1

        summary = format_summary_header(
            raw_count=3,
            deduped_count=len(deduped),
            statuses=statuses,
            ev_cut=0.01,
            conf_gates="default=0.55",
        )

        assert "3" in summary       # raw
        assert "2" in summary       # deduped
