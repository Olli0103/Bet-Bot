"""Tests for data source separation: training data vs live PnL.

Covers:
  1. Dashboard excludes training data from PnL
  2. Importer marks rows as historical_import
  3. Live bets marked as non-training
  4. Backfill marks existing historical rows
  5. Equity curve uses only live data
"""
import importlib
import inspect
import sys

import pytest
from unittest.mock import patch, MagicMock

# Guard: some modules trigger psycopg import at module level
_has_psycopg = importlib.util.find_spec("psycopg") is not None
_skip_no_pg = pytest.mark.skipif(not _has_psycopg, reason="psycopg not installed")


# ── 1) test_dashboard_excludes_training_data_from_pnl ──

class TestDashboardExcludesTrainingData:
    """Verify that dashboard aggregates only include non-training bets."""

    def test_model_has_training_data_columns(self):
        """PlacedBet model must have is_training_data and data_source columns."""
        from src.data.models import PlacedBet

        assert hasattr(PlacedBet, "is_training_data")
        assert hasattr(PlacedBet, "data_source")

        col_td = PlacedBet.__table__.columns["is_training_data"]
        assert col_td.server_default.arg == "false"

        col_ds = PlacedBet.__table__.columns["data_source"]
        assert col_ds.server_default.arg == "live_trade"

    @_skip_no_pg
    def test_bankroll_excludes_training_data(self):
        """BankrollManager.get_current_bankroll must filter is_training_data."""
        from src.core.bankroll import BankrollManager

        source = inspect.getsource(BankrollManager.get_current_bankroll)
        assert "is_training_data" in source

    @_skip_no_pg
    def test_performance_monitor_excludes_training_data(self):
        """PerformanceMonitor queries must filter is_training_data."""
        from src.core.performance_monitor import PerformanceMonitor

        source = inspect.getsource(PerformanceMonitor.get_recent_performance)
        assert "is_training_data" in source or "_LIVE_ONLY" in source


# ── 2) test_importer_marks_historical_as_training_data ──

class TestImporterMarksHistorical:
    """Verify the import script marks all rows as training data."""

    @_skip_no_pg
    def test_historical_bet_helper_sets_flags(self):
        """The _historical_bet helper must set is_training_data=True."""
        sys.path.insert(0, ".")
        from scripts.import_historical_results import _historical_bet

        bet = _historical_bet(
            event_id="test123",
            sport="soccer_epl",
            market="h2h",
            selection="Arsenal",
            odds=2.0,
            stake=1.0,
            status="won",
            pnl=1.0,
        )

        assert bet.is_training_data is True
        assert bet.data_source == "historical_import"
        assert "source=historical_import" in (bet.notes or "")

    @_skip_no_pg
    def test_historical_bet_preserves_notes(self):
        """Existing notes should be preserved when adding source tag."""
        sys.path.insert(0, ".")
        from scripts.import_historical_results import _historical_bet

        bet = _historical_bet(
            event_id="test123",
            sport="soccer_epl",
            market="h2h",
            selection="Arsenal",
            odds=2.0,
            stake=1.0,
            status="won",
            pnl=1.0,
            notes="custom note",
        )

        assert "custom note" in bet.notes
        assert "source=historical_import" in bet.notes

    def test_importer_uses_historical_bet_helper(self):
        """All db.add calls in importer must use _historical_bet, not PlacedBet."""
        import re
        from pathlib import Path

        source = Path("scripts/import_historical_results.py").read_text()
        # Should NOT have db.add(PlacedBet( but should have db.add(_historical_bet(
        raw_inserts = re.findall(r"db\.add\(PlacedBet\(", source)
        historical_inserts = re.findall(r"db\.add\(_historical_bet\(", source)

        assert len(raw_inserts) == 0, f"Found {len(raw_inserts)} raw PlacedBet inserts"
        assert len(historical_inserts) > 0, "No _historical_bet calls found"


# ── 3) test_live_bets_marked_non_training ──

class TestLiveBetsNonTraining:
    """Verify live/ghost/paper bets are correctly marked."""

    @_skip_no_pg
    def test_ghost_trade_marked_live(self):
        """Ghost trading PlacedBet creation must set is_training_data=False."""
        from src.core.ghost_trading import auto_place_virtual_bets

        source = inspect.getsource(auto_place_virtual_bets)
        assert 'data_source="live_trade"' in source
        assert "is_training_data=False" in source

    @_skip_no_pg
    def test_paper_signal_marked_paper(self):
        """Paper signal persistence must set data_source='paper_signal'."""
        from src.core.paper_signals import _persist_paper_signal

        source = inspect.getsource(_persist_paper_signal)
        assert 'data_source="paper_signal"' in source
        assert "is_training_data=False" in source

    @_skip_no_pg
    def test_manual_bet_marked_manual(self):
        """Manual Telegram bet must set data_source='manual'."""
        from src.bot.handlers import _sync_place_bet

        source = inspect.getsource(_sync_place_bet)
        assert 'data_source="manual"' in source


# ── 4) test_backfill_marks_existing_historical_rows ──

class TestBackfillMarksHistorical:
    """Verify backfill script logic."""

    @_skip_no_pg
    def test_backfill_dry_run_returns_counts(self):
        """Backfill in dry-run mode should return count dict without modifying DB."""
        from scripts.backfill_data_source_flags import backfill

        with patch("scripts.backfill_data_source_flags.SessionLocal") as mock_sess:
            mock_db = MagicMock()
            mock_sess.return_value.__enter__ = MagicMock(return_value=mock_db)
            mock_sess.return_value.__exit__ = MagicMock(return_value=False)

            mock_db.scalar.return_value = 0
            mock_db.execute.return_value.all.return_value = []

            result = backfill(dry_run=True)

            assert "historical_import_by_notes" in result
            assert "paper_signal_by_notes" in result
            assert "historical_import_by_heuristic" in result
            assert "total_marked" in result
            mock_db.commit.assert_not_called()

    @_skip_no_pg
    def test_backfill_script_importable(self):
        """The backfill script should be importable without errors."""
        sys.path.insert(0, ".")
        import scripts.backfill_data_source_flags as bsf

        assert hasattr(bsf, "backfill")
        assert hasattr(bsf, "main")


# ── 5) test_equity_curve_uses_only_live_data ──

class TestEquityCurveUsesLiveData:
    """Verify equity curve only includes non-training bets."""

    @_skip_no_pg
    def test_dashboard_data_filters_equity_curve(self):
        """_sync_get_dashboard_data must filter equity_rows by is_training_data."""
        from src.bot.handlers import _sync_get_dashboard_data

        source = inspect.getsource(_sync_get_dashboard_data)
        assert "is_training_data" in source or "live_filter" in source

    @_skip_no_pg
    def test_learning_health_uses_column_filter(self):
        """learning_health(live_only=True) must use is_training_data column."""
        from src.core.learning_monitor import learning_health

        source = inspect.getsource(learning_health)
        assert "is_training_data" in source

    @_skip_no_pg
    def test_training_data_stats_exists(self):
        """training_data_stats function must exist for separate training view."""
        from src.core.learning_monitor import training_data_stats

        assert callable(training_data_stats)

    def test_alembic_migration_exists(self):
        """Migration for data_source columns must exist."""
        from pathlib import Path

        migration = Path("alembic/versions/e7f4a28d1c03_add_data_source_separation_columns.py")
        assert migration.exists(), "Alembic migration file missing"

        content = migration.read_text()
        assert "is_training_data" in content
        assert "data_source" in content
        assert "ix_placed_bets_data_source" in content
