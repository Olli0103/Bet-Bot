"""Tests for src/core/source_health.py — per-source circuit breakers."""
from unittest.mock import patch, MagicMock

import pytest

from src.core.source_health import (
    record_success,
    record_failure,
    is_available,
    get_health_report,
    get_all_health,
)


class TestRecordSuccess:
    @patch("src.core.source_health.cache")
    def test_resets_failures(self, mock_cache):
        mock_cache.get_json.return_value = {"failures": 3, "status": "degraded"}
        record_success("odds_api")
        call_args = mock_cache.set_json.call_args
        saved = call_args[0][1]
        assert saved["failures"] == 0
        assert saved["status"] == "healthy"
        assert saved["total_success"] == 1


class TestRecordFailure:
    @patch("src.core.source_health.cache")
    def test_increments_failures(self, mock_cache):
        mock_cache.get_json.return_value = {"failures": 0}
        record_failure("odds_api", "timeout")
        call_args = mock_cache.set_json.call_args
        saved = call_args[0][1]
        assert saved["failures"] == 1
        assert saved["last_error"] == "timeout"

    @patch("src.core.source_health.cache")
    def test_trips_breaker(self, mock_cache):
        mock_cache.get_json.return_value = {"failures": 4}
        record_failure("odds_api", "500 error")
        call_args = mock_cache.set_json.call_args
        saved = call_args[0][1]
        assert saved["failures"] == 5
        assert saved["status"] == "open"

    @patch("src.core.source_health.cache")
    def test_degraded_status(self, mock_cache):
        # 2 failures for odds_api (max_failures=5, degraded at 2+)
        mock_cache.get_json.return_value = {"failures": 1}
        record_failure("odds_api")
        call_args = mock_cache.set_json.call_args
        saved = call_args[0][1]
        assert saved["failures"] == 2
        assert saved["status"] == "degraded"


class TestIsAvailable:
    @patch("src.core.source_health.cache")
    def test_available_when_no_data(self, mock_cache):
        mock_cache.get_json.return_value = None
        assert is_available("odds_api") is True

    @patch("src.core.source_health.cache")
    def test_available_when_healthy(self, mock_cache):
        mock_cache.get_json.return_value = {"status": "healthy", "failures": 0}
        assert is_available("odds_api") is True

    @patch("src.core.source_health.cache")
    def test_unavailable_when_open(self, mock_cache):
        import time
        mock_cache.get_json.return_value = {
            "status": "open",
            "breaker_tripped_at": time.time(),  # just tripped
        }
        assert is_available("odds_api") is False

    @patch("src.core.source_health.cache")
    @patch("src.core.source_health.time")
    def test_available_after_cooldown(self, mock_time, mock_cache):
        mock_time.time.return_value = 1000000
        mock_cache.get_json.return_value = {
            "status": "open",
            "breaker_tripped_at": 999000,  # 1000 seconds ago (cooldown=300)
        }
        assert is_available("odds_api") is True


class TestGetHealthReport:
    @patch("src.core.source_health.cache")
    def test_generates_report(self, mock_cache):
        mock_cache.get_json.return_value = {
            "status": "healthy",
            "failures": 0,
            "total_success": 100,
            "total_failures": 2,
        }
        report = get_health_report()
        assert "Data Source Health" in report
        assert "The Odds API" in report


class TestGetAllHealth:
    @patch("src.core.source_health.cache")
    def test_returns_all_sources(self, mock_cache):
        mock_cache.get_json.return_value = {"status": "healthy"}
        health = get_all_health()
        assert "odds_api" in health
        assert "thesportsdb" in health
        assert "football_data" in health
