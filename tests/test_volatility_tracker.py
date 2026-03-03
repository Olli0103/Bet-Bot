"""Tests for src/core/volatility_tracker.py — steam move detection.

These tests mock Redis to avoid external dependencies.
"""
import math
from unittest.mock import patch, MagicMock

import pytest

from src.core.volatility_tracker import (
    _cache_key,
    detect_steam_move,
    get_volatility,
    get_volatility_features,
    record_odds_snapshot,
)


class TestDetectSteamMove:
    """Pure function — no Redis dependency."""

    def test_large_movement_no_history(self):
        # No volatility data → threshold = 5% implied prob shift
        assert detect_steam_move(1.8, 2.0, team_volatility=0.0) is True
        # 1/1.8 - 1/2.0 = 0.5556 - 0.5 = 0.0556 > 0.05

    def test_small_movement_no_history(self):
        assert detect_steam_move(1.95, 2.0, team_volatility=0.0) is False
        # 1/1.95 - 1/2.0 = 0.5128 - 0.5 = 0.0128 < 0.05

    def test_movement_exceeds_volatility(self):
        # 2x volatility threshold
        # movement = |1/1.7 - 1/2.0| = |0.5882 - 0.5| = 0.0882
        # threshold = 0.03 * 2 = 0.06
        assert detect_steam_move(1.7, 2.0, team_volatility=0.03) is True

    def test_movement_within_volatility(self):
        # movement = |1/1.95 - 1/2.0| = 0.0128
        # threshold = 0.03 * 2 = 0.06
        assert detect_steam_move(1.95, 2.0, team_volatility=0.03) is False

    def test_invalid_odds(self):
        assert detect_steam_move(1.0, 2.0, team_volatility=0.0) is False
        assert detect_steam_move(2.0, 1.0, team_volatility=0.0) is False
        assert detect_steam_move(0.5, 2.0, team_volatility=0.0) is False

    def test_custom_threshold(self):
        # With 3x multiplier, same movement should not trigger
        # movement = 0.0882, threshold = 0.03 * 3 = 0.09
        assert detect_steam_move(1.7, 2.0, team_volatility=0.03,
                                 threshold_multiplier=3.0) is False


class TestCacheKey:
    def test_normalizes_team_name(self):
        key = _cache_key("Los Angeles Lakers")
        assert key == "volatility:losangeleslakers"

    def test_special_chars_stripped(self):
        # ü is alphanumeric in Python (isalnum()=True), so it's kept
        key = _cache_key("FC Bayern München")
        assert key == "volatility:fcbayernmünchen"


class TestGetVolatility:
    @patch("src.core.volatility_tracker.cache")
    def test_insufficient_data(self, mock_cache):
        mock_cache.get_json.return_value = {"snapshots": [0.5, 0.52]}
        assert get_volatility("TestTeam") == 0.0

    @patch("src.core.volatility_tracker.cache")
    def test_constant_odds(self, mock_cache):
        mock_cache.get_json.return_value = {"snapshots": [0.5, 0.5, 0.5, 0.5]}
        assert get_volatility("TestTeam") == pytest.approx(0.0)

    @patch("src.core.volatility_tracker.cache")
    def test_volatile_odds(self, mock_cache):
        # Alternating: 0.5, 0.6, 0.5, 0.6 → diffs = [0.1, -0.1, 0.1]
        mock_cache.get_json.return_value = {"snapshots": [0.5, 0.6, 0.5, 0.6]}
        vol = get_volatility("TestTeam")
        # mean = (0.1 - 0.1 + 0.1) / 3 ≈ 0.0333
        # var = ((0.1-0.0333)^2 + (-0.1-0.0333)^2 + (0.1-0.0333)^2) / 3
        assert vol > 0.05  # should be a positive, meaningful volatility

    @patch("src.core.volatility_tracker.cache")
    def test_no_cache_data(self, mock_cache):
        mock_cache.get_json.return_value = None
        assert get_volatility("TestTeam") == 0.0


class TestRecordOddsSnapshot:
    @patch("src.core.volatility_tracker.cache")
    def test_appends_snapshot(self, mock_cache):
        mock_cache.get_json.return_value = {"snapshots": [0.5, 0.52]}
        record_odds_snapshot("TestTeam", 0.55)
        call_args = mock_cache.set_json.call_args
        saved_data = call_args[0][1]
        assert saved_data["snapshots"] == [0.5, 0.52, 0.55]

    @patch("src.core.volatility_tracker.cache")
    def test_caps_at_50(self, mock_cache):
        mock_cache.get_json.return_value = {"snapshots": list(range(50))}
        record_odds_snapshot("TestTeam", 999)
        call_args = mock_cache.set_json.call_args
        saved = call_args[0][1]["snapshots"]
        assert len(saved) == 50
        assert saved[-1] == 999
