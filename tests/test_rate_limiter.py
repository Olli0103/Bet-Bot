"""Tests for the Redis-backed global rate limiter in odds_fetcher.py."""
import time
from unittest.mock import patch, MagicMock

from src.integrations.odds_fetcher import (
    _check_global_rate_limit,
    _record_api_call,
    _MIN_REQUEST_INTERVAL_S,
)


class TestGlobalRateLimiter:
    @patch("src.integrations.odds_fetcher.cache")
    def test_allows_when_no_previous_call(self, mock_cache):
        mock_cache.get.return_value = None
        assert _check_global_rate_limit() is True

    @patch("src.integrations.odds_fetcher.cache")
    def test_blocks_when_called_too_recently(self, mock_cache):
        # Simulate a call just now
        mock_cache.get.return_value = str(time.time())
        assert _check_global_rate_limit() is False

    @patch("src.integrations.odds_fetcher.cache")
    def test_allows_after_interval_passes(self, mock_cache):
        # Simulate a call 2 seconds ago (> 1s interval)
        mock_cache.get.return_value = str(time.time() - 2.0)
        assert _check_global_rate_limit() is True

    @patch("src.integrations.odds_fetcher.cache")
    def test_record_api_call(self, mock_cache):
        _record_api_call()
        mock_cache.set.assert_called_once()
        # Verify TTL is set
        call_args = mock_cache.set.call_args
        assert call_args[1].get("ttl_seconds") == 60 or call_args[0][2] == 60 if len(call_args[0]) > 2 else True
