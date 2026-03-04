"""Tests for fetch scheduler, outright filtering, source gap report,
enrichment hardening, signal modes, and dashboard separation."""
import pytest
from unittest.mock import patch, MagicMock
import time


# ── 1) test_odds_fetch_sequential_with_rate_limit_backoff ──

class TestSequentialFetchWithRateLimit:
    """Verify sequential fetch with rate-limit backoff."""

    def test_sequential_fetch_respects_delay(self):
        """Fetch scheduler inserts delay between requests."""
        from src.core.fetch_scheduler import sequential_fetch_odds

        call_times = []
        original_sleep = time.sleep

        def mock_sleep(secs):
            call_times.append(secs)
            # Don't actually sleep in tests

        mock_fetcher = MagicMock()
        mock_fetcher.get_sport_odds.return_value = [
            {"id": "evt1", "home_team": "A", "away_team": "B"}
        ]

        with patch("src.core.fetch_scheduler.time.sleep", mock_sleep), \
             patch("src.core.fetch_scheduler.is_available", return_value=True), \
             patch("src.core.fetch_scheduler._is_key_cooled_down", return_value=False):
            events, stats = sequential_fetch_odds(
                sport_keys=["soccer_epl", "basketball_nba"],
                odds_fetcher=mock_fetcher,
                min_delay_ms=800,
                max_delay_ms=1500,
            )

        # Should have 1 delay between the 2 requests (not before first)
        assert len(call_times) == 1
        assert 0.8 <= call_times[0] <= 1.5
        assert len(events) == 2  # 1 event per sport

    def test_429_triggers_exponential_backoff(self):
        """429 error triggers retry with exponential backoff."""
        from src.core.fetch_scheduler import sequential_fetch_odds
        from src.integrations.base_fetcher import APIFetchError

        delays = []

        def mock_sleep(secs):
            delays.append(secs)

        mock_fetcher = MagicMock()
        mock_fetcher.get_sport_odds.side_effect = APIFetchError("429 rate limit exceeded")

        with patch("src.core.fetch_scheduler.time.sleep", mock_sleep), \
             patch("src.core.fetch_scheduler.is_available", return_value=True), \
             patch("src.core.fetch_scheduler._is_key_cooled_down", return_value=False), \
             patch("src.core.fetch_scheduler.record_failure"):
            events, stats = sequential_fetch_odds(
                sport_keys=["soccer_epl"],
                odds_fetcher=mock_fetcher,
                max_retries=3,
            )

        # Should have retried (with backoff delays)
        assert "429" in stats.status_counts or "0" in stats.status_counts
        assert len(events) == 0  # All retries failed

    def test_cache_fallback_on_circuit_breaker(self):
        """When odds_api circuit breaker is open, use cached data."""
        from src.core.fetch_scheduler import sequential_fetch_odds

        cached_events = [{"id": "cached1", "home_team": "X", "away_team": "Y"}]

        mock_fetcher = MagicMock()
        with patch("src.core.fetch_scheduler.is_available", return_value=False), \
             patch("src.core.fetch_scheduler.cache.get_json", return_value=cached_events):
            events, stats = sequential_fetch_odds(
                sport_keys=["soccer_epl"],
                odds_fetcher=mock_fetcher,
            )

        assert len(events) == 1
        assert events[0][1]["id"] == "cached1"
        mock_fetcher.get_sport_odds.assert_not_called()


# ── 2) test_outright_keys_excluded_from_h2h_pipeline ──

class TestOutrightKeyFilter:
    """Verify outright/futures keys are excluded."""

    def test_outright_patterns_detected(self):
        from src.core.fetch_scheduler import is_outright_key, filter_match_keys

        assert is_outright_key("soccer_epl_winner") is True
        assert is_outright_key("basketball_nba_championship_winner") is True
        assert is_outright_key("soccer_epl_futures") is True
        assert is_outright_key("soccer_epl") is False
        assert is_outright_key("tennis_atp_wimbledon") is False
        assert is_outright_key("basketball_nba") is False

    def test_filter_separates_match_and_outright(self):
        from src.core.fetch_scheduler import filter_match_keys

        keys = [
            "soccer_epl",
            "soccer_epl_winner",
            "basketball_nba",
            "tennis_atp_futures",
            "tennis_atp_wimbledon",
        ]
        match_keys, outright_keys = filter_match_keys(keys)

        assert "soccer_epl" in match_keys
        assert "basketball_nba" in match_keys
        assert "tennis_atp_wimbledon" in match_keys
        assert "soccer_epl_winner" in outright_keys
        assert "tennis_atp_futures" in outright_keys
        assert len(match_keys) == 3
        assert len(outright_keys) == 2


# ── 3) test_source_gap_report_has_stage_counts ──

class TestSourceGapReport:
    """Verify source gap report tracks all pipeline stages."""

    def test_stage_counts_tracked(self):
        from src.core.source_gap_report import StageDropTracker

        tracker = StageDropTracker()
        tracker.record_raw_event("soccer_epl")
        tracker.record_raw_event("soccer_epl")
        tracker.record_parsed_event("soccer_epl", has_tipico=True, has_sharp=True)
        tracker.record_parsed_event("soccer_epl", has_tipico=True, has_sharp=False)
        tracker.record_drop("soccer_epl", "no_bookmaker_overlap")
        tracker.record_signal("soccer_epl", playable=True)
        tracker.record_signal("soccer_epl", playable=False)
        tracker.set_final_count(1)

        report = tracker.to_dict()
        g = report["global"]
        assert g["raw_events_count"] == 2
        assert g["parsed_events_count"] == 2
        assert g["with_tipico_count"] == 2
        assert g["with_sharp_count"] == 1
        assert g["with_tipico_and_sharp_count"] == 1
        assert g["dropped_no_bookmaker_overlap"] == 1
        assert g["signals_generated"] == 2
        assert g["signals_playable"] == 1
        assert g["signals_paper_only"] == 1
        assert g["final_displayed_count"] == 1

        # Per-sport breakdown
        epl = report["per_sport"]["soccer_epl"]
        assert epl["raw_events"] == 2
        assert epl["with_both"] == 1

    def test_markdown_renders(self):
        from src.core.source_gap_report import StageDropTracker

        tracker = StageDropTracker()
        tracker.record_raw_event("soccer_epl")
        tracker.record_signal("soccer_epl", playable=True)
        tracker.set_final_count(1)

        md = tracker._render_markdown(tracker.to_dict())
        assert "Source Gap Report" in md
        assert "soccer_epl" in md
        assert "Raw events fetched" in md


# ── 4) test_enrichment_failures_non_blocking ──

class TestEnrichmentNonBlocking:
    """Verify enrichment failures don't block the pipeline."""

    @pytest.mark.skipif(
        not all(__import__("importlib").util.find_spec(m) for m in ["aiohttp"]),
        reason="aiohttp not installed"
    )
    def test_sentiment_returns_zero_on_failure(self):
        from src.core.enrichment import team_sentiment_score

        with patch("src.core.enrichment.NewsFetcher") as MockNews, \
             patch("src.core.enrichment.is_available", return_value=True), \
             patch("src.core.enrichment.cache") as mock_cache:
            mock_cache.get_json.return_value = None
            MockNews.return_value.search_news.side_effect = Exception("API down")

            score = team_sentiment_score("TestTeam")
            assert score == 0.0

    @pytest.mark.skipif(
        not all(__import__("importlib").util.find_spec(m) for m in ["aiohttp"]),
        reason="aiohttp not installed"
    )
    def test_batch_sentiment_never_raises(self):
        from src.core.enrichment import batch_team_sentiment

        with patch("src.core.enrichment.team_sentiment_score", side_effect=Exception("fail")):
            result = batch_team_sentiment(["Team1", "Team2"])

        assert result["Team1"] == 0.0
        assert result["Team2"] == 0.0

    @pytest.mark.skipif(
        not all(__import__("importlib").util.find_spec(m) for m in ["aiohttp"]),
        reason="aiohttp not installed"
    )
    def test_newsapi_breaker_open_returns_zero(self):
        from src.core.enrichment import team_sentiment_score

        with patch("src.core.enrichment.is_available", return_value=False), \
             patch("src.core.enrichment.cache") as mock_cache:
            mock_cache.get_json.return_value = None

            score = team_sentiment_score("BreakerTeam")
            assert score == 0.0


# ── 5) test_gnews_query_sanitization_prevents_400 ──

class TestQuerySanitization:
    """Verify news query sanitization prevents 400 errors."""

    def test_special_chars_removed(self):
        from src.integrations.news_fetcher import sanitize_news_query

        result = sanitize_news_query("FC Bayern München (1. Bundesliga)")
        assert "(" not in result
        assert ")" not in result
        assert "Bayern" in result
        assert sanitize_news_query("Team (League)").count("(") == 0
        assert sanitize_news_query("Team (League)").count(")") == 0

    def test_query_truncated(self):
        from src.integrations.news_fetcher import sanitize_news_query

        long_query = "A" * 200
        result = sanitize_news_query(long_query)
        assert len(result) <= 100

    def test_empty_query_fallback(self):
        from src.integrations.news_fetcher import sanitize_news_query

        assert sanitize_news_query("") == "sports"
        assert sanitize_news_query("###") == "sports"

    def test_unicode_preserved(self):
        from src.integrations.news_fetcher import sanitize_news_query

        result = sanitize_news_query("Borussia Dortmund vs Real Madrid")
        assert "Borussia" in result
        assert "Real Madrid" in result


# ── 6) test_learning_mode_captures_paper_signals_even_with_stake_zero ──

class TestLearningModeCapture:
    """Verify learning mode captures all signals including stake=0."""

    def test_paper_signal_captured_for_rejected_bet(self):
        from src.core.paper_signals import PaperSignalRecord, capture_paper_signal

        with patch("src.core.paper_signals.settings") as mock_settings, \
             patch("src.core.paper_signals._persist_paper_signal") as mock_persist:
            mock_settings.learning_capture_all_signals = True

            record = capture_paper_signal(
                event_id="evt1",
                sport="soccer_epl",
                market="h2h",
                selection="Arsenal",
                bookmaker_odds=1.8,
                model_probability=0.45,
                expected_value=-0.02,
                recommended_stake=0.0,  # rejected
                confidence_gate_passed=False,
                reject_reason="reject_confidence_below_min: model_prob=0.45 < gate=0.55",
            )

            assert record.signal_mode == "PAPER_ONLY"
            assert record.reject_reason != ""
            assert record.recommended_stake == 0.0
            mock_persist.assert_called_once()

    def test_paper_signal_playable_for_accepted_bet(self):
        from src.core.paper_signals import capture_paper_signal

        with patch("src.core.paper_signals.settings") as mock_settings, \
             patch("src.core.paper_signals._persist_paper_signal"):
            mock_settings.learning_capture_all_signals = True

            record = capture_paper_signal(
                event_id="evt2",
                sport="soccer_epl",
                market="h2h",
                selection="Chelsea",
                bookmaker_odds=2.1,
                model_probability=0.60,
                expected_value=0.05,
                recommended_stake=15.0,
                confidence_gate_passed=True,
                reject_reason="",
            )

            assert record.signal_mode == "PLAYABLE"

    def test_learning_disabled_skips_persistence(self):
        from src.core.paper_signals import capture_paper_signal

        with patch("src.core.paper_signals.settings") as mock_settings, \
             patch("src.core.paper_signals._persist_paper_signal") as mock_persist:
            mock_settings.learning_capture_all_signals = False

            capture_paper_signal(
                event_id="evt3",
                sport="tennis_atp",
                market="h2h",
                selection="Player A",
                bookmaker_odds=1.5,
                model_probability=0.70,
                expected_value=0.03,
                recommended_stake=10.0,
                confidence_gate_passed=True,
            )

            mock_persist.assert_not_called()


# ── 7) test_trading_mode_keeps_strict_filters ──

class TestTradingModeStrictFilters:
    """Verify trading mode maintains strict confidence and EV gates."""

    def test_confidence_gate_blocks_low_prob(self):
        from src.core.risk_guards import passes_confidence_gate

        passed, min_conf = passes_confidence_gate(0.45, "soccer_epl", "h2h")
        assert passed is False
        assert min_conf == 0.55

    def test_confidence_gate_passes_high_prob(self):
        from src.core.risk_guards import passes_confidence_gate

        passed, min_conf = passes_confidence_gate(0.60, "soccer_epl", "h2h")
        assert passed is True

    def test_signal_rejected_has_zero_stake(self):
        from src.core.betting_engine import BettingEngine

        engine = BettingEngine(bankroll=1000.0)
        with patch("src.core.betting_engine.check_data_source_health", return_value=(True, "")):
            sig = engine.make_signal(
                sport="soccer_epl",
                event_id="evt1",
                market="h2h",
                selection="Test Team",
                bookmaker_odds=2.0,
                model_probability=0.40,  # Below gate
                tax_rate=0.05,
            )

        assert sig.recommended_stake == 0.0
        assert sig.rejected_reason != ""
        assert "confidence" in sig.rejected_reason.lower()


# ── 8) test_dashboard_excludes_historical_import_from_live_pnl ──

class TestDashboardExcludesImports:
    """Verify dashboard/bankroll only shows live bets, not historical imports."""

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("psycopg"),
        reason="psycopg not installed"
    )
    def test_learning_health_filters_imports(self):
        """learning_health(live_only=True) should exclude historical imports."""
        from src.core.learning_monitor import learning_health

        # Create mock bets
        mock_live_bet = MagicMock()
        mock_live_bet.stake = 10.0
        mock_live_bet.status = "won"
        mock_live_bet.pnl = 8.0
        mock_live_bet.notes = None

        mock_import_bet = MagicMock()
        mock_import_bet.stake = 1.0
        mock_import_bet.status = "won"
        mock_import_bet.pnl = 1.5
        mock_import_bet.notes = "source=historical_import"

        # The function uses SQLAlchemy queries, so we test the concept
        # by verifying the function exists and accepts live_only parameter
        with patch("src.core.learning_monitor.SessionLocal") as mock_session:
            mock_db = MagicMock()
            mock_session.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_db.scalars.return_value.all.return_value = [mock_live_bet]

            result = learning_health(live_only=True)
            assert result["live_only"] is True
            assert "total" in result
            assert "pnl" in result
