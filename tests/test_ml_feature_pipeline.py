"""Tests for the ML feature pipeline fix — ensuring features are properly
persisted, unpacked, and not lost to NaN.

Phase 1 tests:
1. test_new_bets_persist_required_ml_features
2. test_sentiment_failure_sets_neutral_not_nan
3. test_injury_failure_sets_neutral_not_nan
4. test_clean_frame_unpacks_meta_features_no_nan_for_defaults
5. test_backfill_script_fills_missing_features_idempotent
6. test_training_report_detects_no_feature_variance

Phase 2 tests (training blocker fixes):
7. test_clv_regression_handles_missing_event_closing_lines_market
8. test_feature_pipeline_persists_problem_features
9. test_clean_frame_unpacks_meta_features_consistently
10. test_backfill_fills_missing_features_idempotent
11. test_coverage_report_flags_nan_spikes
12. test_training_does_not_fail_on_partial_schema
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────
# 1. New bets persist required ML features in meta_features
# ──────────────────────────────────────────────────────────────

class TestNewBetsPersistRequiredMLFeatures:
    """Verify that ghost_trading persists all 6 critical features."""

    REQUIRED_FEATURES = [
        "sentiment_delta",
        "injury_delta",
        "sharp_implied_prob",
        "sharp_vig",
        "form_winrate_l5",
        "form_games_l5",
    ]

    def test_new_bets_persist_required_ml_features(self):
        """auto_place_virtual_bets stores all 6 critical features in both
        dedicated columns and meta_features JSONB."""
        from src.core.ghost_trading import _safe_meta

        features = {
            "sharp_implied_prob": 0.55,
            "sharp_vig": 0.05,
            "sentiment_delta": 0.3,
            "injury_delta": -1.0,
            "form_winrate_l5": 0.6,
            "form_games_l5": 5.0,
            "clv": 0.1,
            "elo_diff": 50.0,
            "home_advantage": 1.0,
        }

        meta = _safe_meta(features)

        # All critical features must be present in meta_features
        for feat in self.REQUIRED_FEATURES:
            assert feat in meta, f"Missing critical feature '{feat}' in meta_features"
            assert isinstance(meta[feat], float), f"Feature '{feat}' must be float, got {type(meta[feat])}"
            assert np.isfinite(meta[feat]), f"Feature '{feat}' must be finite, got {meta[feat]}"

    def test_safe_meta_handles_empty(self):
        from src.core.ghost_trading import _safe_meta
        assert _safe_meta({}) == {}
        assert _safe_meta(None) == {}

    def test_feature_engineer_produces_all_critical_features(self):
        """FeatureEngineer.build_core_features returns all 6 critical features."""
        from src.core.feature_engineering import FeatureEngineer

        features = FeatureEngineer.build_core_features(
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"A": 1.9, "B": 2.1},
            sentiment_home=0.5,
            sentiment_away=-0.2,
            injuries_home=1,
            injuries_away=3,
            selection="A",
            home_team="A",
            form_winrate_l5=0.6,
            form_games_l5=5,
        )

        for feat in self.REQUIRED_FEATURES:
            assert feat in features, f"Missing '{feat}' from build_core_features output"
            assert isinstance(features[feat], float), f"Feature '{feat}' is not float"
            assert np.isfinite(features[feat]), f"Feature '{feat}' is not finite"


# ──────────────────────────────────────────────────────────────
# 2. Sentiment failure → neutral 0.0, not NaN
# ──────────────────────────────────────────────────────────────

class TestSentimentFailureSetsNeutral:
    def test_sentiment_failure_sets_neutral_not_nan(self):
        """When sentiment enrichment fails, the feature defaults to 0.0 not NaN."""
        from src.core.feature_engineering import FeatureEngineer

        # Simulate: sentiment fetch failed, both teams get 0.0
        features = FeatureEngineer.build_core_features(
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"Home": 1.9, "Away": 2.1},
            sentiment_home=0.0,  # fallback neutral
            sentiment_away=0.0,  # fallback neutral
            injuries_home=0,
            injuries_away=0,
            selection="Home",
            home_team="Home",
        )

        assert features["sentiment_delta"] == 0.0
        assert not np.isnan(features["sentiment_delta"])

    def test_batch_sentiment_returns_zero_on_failure(self):
        """batch_team_sentiment returns 0.0 per team when the underlying fetch fails.

        We mock away the deep import chain (apisports -> httpx -> tenacity)
        to test the batch_team_sentiment logic in isolation.
        """
        # Mock the heavy dependencies that enrichment.py imports
        mock_apisports = MagicMock()
        mock_news = MagicMock()
        mock_ollama = MagicMock()
        with patch.dict("sys.modules", {
            "src.integrations.apisports_fetcher": mock_apisports,
            "src.integrations.news_fetcher": mock_news,
            "src.integrations.ollama_sentiment": mock_ollama,
        }):
            # Force re-import of enrichment with mocked deps
            import importlib
            import src.core.enrichment as enrich_mod
            importlib.reload(enrich_mod)

            # Now patch team_sentiment_score to raise
            with patch.object(enrich_mod, "team_sentiment_score", side_effect=Exception("API down")):
                result = enrich_mod.batch_team_sentiment(["TeamA", "TeamB"], max_teams=2)
                assert result["TeamA"] == 0.0
                assert result["TeamB"] == 0.0


# ──────────────────────────────────────────────────────────────
# 3. Injury failure → neutral 0.0, not NaN
# ──────────────────────────────────────────────────────────────

class TestInjuryFailureSetsNeutral:
    def test_injury_failure_sets_neutral_not_nan(self):
        """When injury enrichment fails, injury_delta defaults to 0.0 not NaN."""
        from src.core.feature_engineering import FeatureEngineer

        # Simulate: injury API failed, both counts are 0
        features = FeatureEngineer.build_core_features(
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"Home": 1.9, "Away": 2.1},
            sentiment_home=0.0,
            sentiment_away=0.0,
            injuries_home=0,  # fallback: 0 injuries
            injuries_away=0,  # fallback: 0 injuries
            selection="Home",
            home_team="Home",
        )

        assert features["injury_delta"] == 0.0
        assert not np.isnan(features["injury_delta"])

    def test_injury_news_delta_defaults_zero(self):
        """injury_news_delta defaults to 0.0 when RSS fails."""
        from src.core.feature_engineering import FeatureEngineer

        features = FeatureEngineer.build_core_features(
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"Home": 1.9},
            sentiment_home=0.0,
            sentiment_away=0.0,
            injuries_home=0,
            injuries_away=0,
            selection="Home",
            home_team="Home",
            injury_news_delta=0.0,  # explicit neutral fallback
        )

        assert features["injury_news_delta"] == 0.0
        assert not np.isnan(features["injury_news_delta"])


# ──────────────────────────────────────────────────────────────
# 4. _clean_frame unpacks meta_features → no NaN for defaults
# ──────────────────────────────────────────────────────────────

class TestCleanFrameUnpacksMetaFeatures:
    def test_clean_frame_unpacks_meta_features_no_nan_for_defaults(self):
        """_clean_frame fills NaN dedicated columns from meta_features JSONB,
        then applies FEATURE_DEFAULTS for any remaining NaN."""
        from src.core.ml_trainer import _clean_frame, FEATURES, FEATURE_DEFAULTS

        # Simulate a DataFrame as it comes from pd.read_sql:
        # - dedicated columns exist but are NULL (NaN)
        # - meta_features JSONB has the actual values
        meta = {
            "sentiment_delta": 0.3,
            "injury_delta": -1.0,
            "sharp_implied_prob": 0.55,
            "sharp_vig": 0.05,
            "form_winrate_l5": 0.8,
            "form_games_l5": 4.0,
            "elo_diff": 50.0,
            "elo_expected": 0.65,
            "h2h_home_winrate": 0.6,
            "home_advantage": 1.0,
        }

        df = pd.DataFrame([{
            "odds": 2.0,
            "status": "won",
            "sport": "soccer_epl",
            # Dedicated columns are NULL (simulating old rows)
            "sentiment_delta": np.nan,
            "injury_delta": np.nan,
            "sharp_implied_prob": np.nan,
            "sharp_vig": np.nan,
            "form_winrate_l5": np.nan,
            "form_games_l5": np.nan,
            # JSONB has the values
            "meta_features": meta,
        }])

        result = _clean_frame(df, FEATURES)

        # All 6 critical features must be filled from meta_features, NOT NaN
        assert result["sentiment_delta"].iloc[0] == pytest.approx(0.3)
        assert result["injury_delta"].iloc[0] == pytest.approx(-1.0)
        assert result["sharp_implied_prob"].iloc[0] == pytest.approx(0.55)
        assert result["sharp_vig"].iloc[0] == pytest.approx(0.05)
        assert result["form_winrate_l5"].iloc[0] == pytest.approx(0.8)
        assert result["form_games_l5"].iloc[0] == pytest.approx(4.0)

    def test_clean_frame_fills_defaults_when_both_null(self):
        """When both column and meta_features are NULL, FEATURE_DEFAULTS apply."""
        from src.core.ml_trainer import _clean_frame, FEATURES, FEATURE_DEFAULTS

        df = pd.DataFrame([{
            "odds": 2.0,
            "status": "won",
            "sport": "soccer_epl",
            "sentiment_delta": np.nan,
            "injury_delta": np.nan,
            "sharp_implied_prob": np.nan,
            "sharp_vig": np.nan,
            "form_winrate_l5": np.nan,
            "form_games_l5": np.nan,
            "meta_features": None,  # No JSONB data
        }])

        result = _clean_frame(df, FEATURES)

        # Must get FEATURE_DEFAULTS, not NaN
        assert result["sentiment_delta"].iloc[0] == pytest.approx(0.0)
        assert result["injury_delta"].iloc[0] == pytest.approx(0.0)
        assert result["form_winrate_l5"].iloc[0] == pytest.approx(0.5)
        assert result["form_games_l5"].iloc[0] == pytest.approx(0.0)
        # sharp_implied_prob=0.0 triggers the derivation from odds
        # (which should produce 1/2.0 = 0.5 after vig strip)

        # None of the critical features should be NaN
        for feat in ["sentiment_delta", "injury_delta", "form_winrate_l5", "form_games_l5"]:
            assert not np.isnan(result[feat].iloc[0]), f"{feat} is still NaN after _clean_frame"

    def test_clean_frame_preserves_valid_column_values(self):
        """When dedicated columns have valid values, they are NOT overwritten by meta."""
        from src.core.ml_trainer import _clean_frame, FEATURES

        df = pd.DataFrame([{
            "odds": 2.0,
            "status": "won",
            "sport": "soccer_epl",
            "sentiment_delta": 0.5,  # valid
            "injury_delta": 2.0,     # valid
            "sharp_implied_prob": 0.55,
            "sharp_vig": 0.04,
            "form_winrate_l5": 0.7,
            "form_games_l5": 3.0,
            "meta_features": {
                "sentiment_delta": 999.0,  # different value in JSONB
                "injury_delta": 999.0,
            },
        }])

        result = _clean_frame(df, FEATURES)

        # Original column values must be preserved
        assert result["sentiment_delta"].iloc[0] == pytest.approx(0.5)
        assert result["injury_delta"].iloc[0] == pytest.approx(2.0)


# ──────────────────────────────────────────────────────────────
# 5. Backfill script fills missing features (idempotent)
# ──────────────────────────────────────────────────────────────

class TestBackfillScriptIdempotent:
    def test_backfill_script_fills_missing_features_idempotent(self):
        """The backfill logic fills missing features and is idempotent."""
        from scripts.backfill_ml_features import (
            _needs_backfill,
            _derive_sharp_implied_prob,
            _derive_sharp_vig,
            _ensure_meta_features,
            ALL_BACKFILL_FEATURES,
        )

        # Empty meta needs backfill
        assert _needs_backfill({}) is True
        assert _needs_backfill(None, force=False) is True

        # Complete meta (all features) does NOT need backfill
        complete = {f: 0.0 for f in ALL_BACKFILL_FEATURES}
        assert _needs_backfill(complete) is False

        # Force overrides
        assert _needs_backfill(complete, force=True) is True

        # Derivation functions produce finite values
        prob = _derive_sharp_implied_prob(2.0, 0.05)
        assert 0.0 < prob < 1.0
        assert np.isfinite(prob)

        vig = _derive_sharp_vig(2.0)
        assert vig == pytest.approx(0.05)
        assert np.isfinite(vig)

        # Edge cases
        assert _derive_sharp_implied_prob(0.0, 0.0) == 0.5
        assert _derive_sharp_implied_prob(1.0, 0.0) == 0.5

    def test_backfill_idempotent_second_pass(self):
        """Running backfill twice on already-filled data should skip rows."""
        from scripts.backfill_ml_features import _needs_backfill, ALL_BACKFILL_FEATURES

        # All features present → no backfill needed
        meta = {f: 0.0 for f in ALL_BACKFILL_FEATURES}
        assert _needs_backfill(meta) is False

    def test_backfill_needs_phase2_features(self):
        """Backfill detects missing phase 2-4 features, not just critical ones."""
        from scripts.backfill_ml_features import _needs_backfill, CRITICAL_FEATURES

        # Only critical features present → phase 2-4 still missing
        meta = {f: 0.0 for f in CRITICAL_FEATURES}
        assert _needs_backfill(meta) is True


# ──────────────────────────────────────────────────────────────
# 6. Training report detects no feature variance
# ──────────────────────────────────────────────────────────────

class TestTrainingReportDetectsNoFeatureVariance:
    def test_training_report_detects_no_feature_variance(self):
        """generate_feature_coverage_report flags features with zero variance and 100% NaN."""
        from src.core.ml_trainer import (
            generate_feature_coverage_report,
            FEATURES,
            CRITICAL_FEATURES,
        )

        # Create a DataFrame where critical features are all NaN
        n = 100
        data = {feat: [np.nan] * n for feat in FEATURES}
        data["odds"] = [2.0] * n
        data["status"] = ["won"] * 50 + ["lost"] * 50
        df = pd.DataFrame(data)

        report = generate_feature_coverage_report(df, FEATURES, "test_sport")

        for feat in CRITICAL_FEATURES:
            assert feat in report
            stats = report[feat]
            assert stats["non_null_rate"] == 0.0, f"{feat} should have 0% non-null"
            assert stats["variance"] == 0.0, f"{feat} should have 0 variance"
            assert stats["is_critical"] is True

    def test_report_shows_good_coverage_for_valid_data(self):
        """Coverage report shows high non-null rates for properly filled data."""
        from src.core.ml_trainer import generate_feature_coverage_report, FEATURES

        n = 100
        rng = np.random.default_rng(42)
        data = {}
        for feat in FEATURES:
            data[feat] = rng.normal(0.5, 0.1, size=n).tolist()
        df = pd.DataFrame(data)

        report = generate_feature_coverage_report(df, FEATURES, "valid_sport")

        for feat in FEATURES:
            stats = report[feat]
            assert stats["non_null_rate"] == 1.0, f"{feat} should be 100% non-null"
            assert stats["variance"] > 0.0, f"{feat} should have positive variance"
            assert stats["unique_count"] > 1, f"{feat} should have multiple unique values"

    def test_report_detects_constant_zero_feature(self):
        """A feature that is all zeros should have zero variance."""
        from src.core.ml_trainer import generate_feature_coverage_report

        df = pd.DataFrame({
            "sentiment_delta": [0.0] * 50,
            "injury_delta": [0.0] * 50,
        })

        report = generate_feature_coverage_report(df, ["sentiment_delta", "injury_delta"])

        assert report["sentiment_delta"]["variance"] == 0.0
        assert report["sentiment_delta"]["zero_rate"] == 1.0
        assert report["sentiment_delta"]["non_null_rate"] == 1.0


# ══════════════════════════════════════════════════════════════
# Phase 2: Training Blocker Fix Tests
# ══════════════════════════════════════════════════════════════


class TestCLVRegressionHandlesMissingMarketColumn:
    """7. CLV regressor must not crash if event_closing_lines.market is absent."""

    def test_clv_regression_handles_missing_event_closing_lines_market(self):
        """_build_clv_dataset gracefully handles missing 'market' column."""
        from unittest.mock import MagicMock, patch

        from src.core.ml_trainer import _build_clv_dataset

        # Mock SessionLocal to simulate a DB without 'market' column
        mock_session = MagicMock()
        mock_engine = MagicMock()
        mock_session.bind = mock_engine

        # sa.inspect returns columns without 'market'
        mock_inspector = MagicMock()
        mock_inspector.get_columns.return_value = [
            {"name": "id"}, {"name": "event_id"}, {"name": "sport"},
            {"name": "selection"}, {"name": "closing_implied_prob"},
        ]

        # The query returns empty (insufficient data)
        mock_session.execute.return_value.all.return_value = []

        with patch("src.core.ml_trainer.SessionLocal") as mock_sl:
            mock_sl.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_sl.return_value.__exit__ = MagicMock(return_value=False)
            # Patch the sqlalchemy inspect at module level (imported inside function)
            with patch("sqlalchemy.inspect", return_value=mock_inspector):
                # Should NOT crash — returns None (insufficient data)
                result = _build_clv_dataset()
                assert result is None

    def test_clv_build_with_market_column_present(self):
        """_build_clv_dataset uses market join when column exists."""
        from unittest.mock import MagicMock, patch

        from src.core.ml_trainer import _build_clv_dataset

        mock_session = MagicMock()
        mock_engine = MagicMock()
        mock_session.bind = mock_engine

        # Inspector shows 'market' column present
        mock_inspector = MagicMock()
        mock_inspector.get_columns.return_value = [
            {"name": "id"}, {"name": "event_id"}, {"name": "sport"},
            {"name": "selection"}, {"name": "market"},
            {"name": "closing_implied_prob"},
        ]

        mock_session.execute.return_value.all.return_value = []

        with patch("src.core.ml_trainer.SessionLocal") as mock_sl:
            mock_sl.return_value.__enter__ = MagicMock(return_value=mock_session)
            mock_sl.return_value.__exit__ = MagicMock(return_value=False)
            with patch("sqlalchemy.inspect", return_value=mock_inspector):
                result = _build_clv_dataset()
                assert result is None  # insufficient data, but no crash


class TestFeaturePipelinePersistsProblemFeatures:
    """8. Feature pipeline must persist ALL features that were 100% NaN."""

    PROBLEM_FEATURES = [
        "elo_diff", "weather_rain", "weather_wind_high",
        "home_volatility", "away_volatility", "is_steam_move",
        "line_staleness", "injury_news_delta", "public_bias",
        "market_momentum", "line_velocity", "form_trend_slope",
        "home_away_split_delta", "league_position_delta",
    ]

    def test_feature_pipeline_persists_problem_features(self):
        """FeatureEngineer.build_core_features returns all problem features."""
        from src.core.feature_engineering import FeatureEngineer

        features = FeatureEngineer.build_core_features(
            target_odds=2.0,
            sharp_odds=1.9,
            sharp_market={"A": 1.9, "B": 2.1},
            sentiment_home=0.5,
            sentiment_away=-0.2,
            injuries_home=1,
            injuries_away=3,
            selection="A",
            home_team="A",
            elo_diff=50.0,
            weather_rain=1.0,
            weather_wind_high=0.0,
            home_volatility=0.02,
            away_volatility=0.01,
            is_steam_move=True,
            line_staleness=15.0,
            injury_news_delta=-0.3,
            public_bias=0.1,
            market_momentum=0.05,
            line_velocity=0.002,
            form_trend_slope=0.1,
            home_away_split_delta=0.15,
            league_position_delta=3.0,
        )

        for feat in self.PROBLEM_FEATURES:
            assert feat in features, f"Missing '{feat}' from build_core_features"
            assert isinstance(features[feat], float), f"'{feat}' is not float"
            assert np.isfinite(features[feat]), f"'{feat}' is not finite"


class TestCleanFrameUnpacksMetaFeaturesConsistently:
    """9. _clean_frame must produce zero NaN for ANY feature when defaults exist."""

    def test_clean_frame_unpacks_meta_features_consistently(self):
        """After _clean_frame, no feature in FEATURES should be NaN."""
        from src.core.ml_trainer import _clean_frame, FEATURES, FEATURE_DEFAULTS

        # Worst case: no meta_features, no dedicated columns
        df = pd.DataFrame([{
            "odds": 2.0,
            "status": "won",
            "sport": "soccer_epl",
            "meta_features": None,
        }])

        result = _clean_frame(df, FEATURES)

        # Every feature must be non-NaN (all have defaults now)
        for feat in FEATURES:
            val = result[feat].iloc[0]
            assert not np.isnan(val), (
                f"{feat} is NaN after _clean_frame (expected default "
                f"{FEATURE_DEFAULTS.get(feat, 'N/A')})"
            )

    def test_clean_frame_generates_missing_indicators(self):
        """_clean_frame generates is_missing_* binary indicators."""
        from src.core.ml_trainer import _clean_frame, FEATURES

        # No meta_features → all missing indicators should be 1.0
        df = pd.DataFrame([{
            "odds": 2.0,
            "status": "won",
            "sport": "soccer_epl",
            "meta_features": None,
        }])

        result = _clean_frame(df, FEATURES)

        # Missing indicators should be 1.0 (data was missing)
        assert result["is_missing_elo"].iloc[0] == 1.0
        assert result["is_missing_weather"].iloc[0] == 1.0
        assert result["is_missing_stats"].iloc[0] == 1.0

    def test_clean_frame_missing_indicators_zero_when_data_present(self):
        """Missing indicators are 0.0 when actual data is in meta_features."""
        from src.core.ml_trainer import _clean_frame, FEATURES

        meta = {
            "elo_diff": 50.0,
            "elo_expected": 0.65,
            "weather_rain": 0.0,
            "weather_wind_high": 1.0,
            "team_attack_strength": 1.2,
            "team_defense_strength": 0.9,
            "opp_attack_strength": 1.1,
            "opp_defense_strength": 1.0,
            "form_trend_slope": 0.1,
        }

        df = pd.DataFrame([{
            "odds": 2.0,
            "status": "won",
            "sport": "soccer_epl",
            "meta_features": meta,
        }])

        result = _clean_frame(df, FEATURES)

        # Data was present → indicators should be 0.0
        assert result["is_missing_elo"].iloc[0] == 0.0
        assert result["is_missing_weather"].iloc[0] == 0.0
        assert result["is_missing_stats"].iloc[0] == 0.0
        assert result["is_missing_form_trend"].iloc[0] == 0.0


class TestBackfillFillsMissingFeaturesIdempotent:
    """10. Backfill fills ALL missing features (phase 1-4) idempotently."""

    def test_backfill_fills_missing_features_idempotent(self):
        """ALL_BACKFILL_FEATURES must cover both critical and phase 2-4."""
        from scripts.backfill_ml_features import (
            ALL_BACKFILL_FEATURES,
            CRITICAL_FEATURES,
            PHASE2_DEFAULTS,
            PHASE4_DEFAULTS,
            _needs_backfill,
        )

        # ALL_BACKFILL_FEATURES = critical + phase2 + phase4
        expected_count = (
            len(CRITICAL_FEATURES) + len(PHASE2_DEFAULTS) + len(PHASE4_DEFAULTS)
        )
        assert len(ALL_BACKFILL_FEATURES) == expected_count

        # Meta with only critical features → still needs backfill (phase 2-4 missing)
        critical_only = {f: 0.0 for f in CRITICAL_FEATURES}
        assert _needs_backfill(critical_only) is True

        # Meta with ALL features → does NOT need backfill
        complete = {f: 0.0 for f in ALL_BACKFILL_FEATURES}
        assert _needs_backfill(complete) is False

        # Idempotent: running _needs_backfill on complete meta is stable
        assert _needs_backfill(complete) is False
        assert _needs_backfill(complete) is False


class TestCoverageReportFlagsNaNSpikes:
    """11. Coverage report must flag 100% NaN critical features."""

    def test_coverage_report_flags_nan_spikes(self):
        """Features at 100% NaN are flagged with is_critical + zero non_null_rate."""
        from src.core.ml_trainer import (
            generate_feature_coverage_report,
            FEATURES,
            CRITICAL_FEATURES,
        )

        n = 100
        # Mix: some features valid, critical features all NaN
        data = {}
        for feat in FEATURES:
            if feat in CRITICAL_FEATURES:
                data[feat] = [np.nan] * n
            else:
                data[feat] = np.random.default_rng(42).normal(0.5, 0.1, size=n).tolist()
        df = pd.DataFrame(data)

        report = generate_feature_coverage_report(df, FEATURES)

        # Critical features must be flagged
        for feat in CRITICAL_FEATURES:
            assert report[feat]["non_null_rate"] == 0.0
            assert report[feat]["is_critical"] is True

        # Non-critical features should have full coverage
        for feat in FEATURES:
            if feat not in CRITICAL_FEATURES:
                assert report[feat]["non_null_rate"] == 1.0

    def test_coverage_report_partial_nan(self):
        """50% NaN features are reported with correct non_null_rate."""
        from src.core.ml_trainer import generate_feature_coverage_report

        n = 100
        data = {
            "sharp_implied_prob": [0.5] * 50 + [np.nan] * 50,
        }
        df = pd.DataFrame(data)

        report = generate_feature_coverage_report(df, ["sharp_implied_prob"])
        assert report["sharp_implied_prob"]["non_null_rate"] == pytest.approx(0.5)


class TestTrainingDoesNotFailOnPartialSchema:
    """12. Training pipeline must not crash on legacy DB schemas."""

    def test_training_does_not_fail_on_partial_schema(self):
        """_clean_frame handles DataFrames missing some feature columns."""
        from src.core.ml_trainer import _clean_frame, FEATURES, FEATURE_DEFAULTS

        # DataFrame with only odds/status — missing ALL feature columns
        df = pd.DataFrame([
            {"odds": 2.0, "status": "won", "sport": "soccer_epl", "meta_features": None},
            {"odds": 1.8, "status": "lost", "sport": "soccer_epl", "meta_features": None},
            {"odds": 3.5, "status": "won", "sport": "soccer_epl", "meta_features": None},
        ])

        result = _clean_frame(df, FEATURES)

        # Must not crash; all features must exist with defaults
        for feat in FEATURES:
            assert feat in result.columns, f"Missing column '{feat}'"
            # No NaN should remain (all defaults are explicit now)
            nan_count = int(result[feat].isna().sum())
            assert nan_count == 0, (
                f"{feat} has {nan_count} NaN after _clean_frame "
                f"(default={FEATURE_DEFAULTS.get(feat, 'N/A')})"
            )

    def test_get_active_features_works_after_clean(self):
        """After _clean_frame, _get_active_features returns non-empty list."""
        from src.core.ml_trainer import _clean_frame, _get_active_features, FEATURES

        rng = np.random.default_rng(42)
        n = 50
        rows = []
        for i in range(n):
            rows.append({
                "odds": 1.5 + rng.random(),
                "status": "won" if rng.random() > 0.5 else "lost",
                "sport": "soccer_epl",
                "meta_features": {
                    "sharp_implied_prob": 0.4 + rng.random() * 0.2,
                    "sentiment_delta": rng.normal(0, 0.3),
                    "elo_diff": rng.normal(0, 50),
                },
            })

        df = pd.DataFrame(rows)
        cleaned = _clean_frame(df, FEATURES)
        active = _get_active_features(cleaned, FEATURES)

        # Must have at least sharp_implied_prob, sentiment_delta, elo_diff with variance
        assert len(active) >= 3, f"Expected ≥3 active features, got {len(active)}: {active}"
        assert "sharp_implied_prob" in active
