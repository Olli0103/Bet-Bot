"""Tests for src/integrations/sportsdb_fetcher.py — TheSportsDB integration."""
from unittest.mock import patch, MagicMock

import pytest

from src.integrations.sportsdb_fetcher import SportsDBFetcher, _safe_int


class TestSafeInt:
    def test_valid_int(self):
        assert _safe_int(42) == 42

    def test_string_int(self):
        assert _safe_int("3") == 3

    def test_none(self):
        assert _safe_int(None) == 0

    def test_invalid(self):
        assert _safe_int("abc") == 0


class TestNormalizeEvent:
    def test_normalizes_complete_event(self):
        raw = {
            "idEvent": "12345",
            "dateEvent": "2026-03-01",
            "strTime": "15:00",
            "strHomeTeam": "Arsenal",
            "strAwayTeam": "Chelsea",
            "intHomeScore": "2",
            "intAwayScore": "1",
            "strLeague": "English Premier League",
            "strSeason": "2025-2026",
            "strVenue": "Emirates Stadium",
            "strStatus": "Match Finished",
            "intHomeShots": "12",
            "intAwayShots": "8",
            "intRound": "23",
        }
        result = SportsDBFetcher._normalize_event(raw)
        assert result["event_id"] == "12345"
        assert result["home_team"] == "Arsenal"
        assert result["away_team"] == "Chelsea"
        assert result["home_score"] == 2
        assert result["away_score"] == 1
        assert result["home_shots"] == 12
        assert result["source"] == "thesportsdb"

    def test_normalizes_incomplete_event(self):
        raw = {"idEvent": "99", "strHomeTeam": "TeamA", "strAwayTeam": "TeamB"}
        result = SportsDBFetcher._normalize_event(raw)
        assert result["home_score"] is None
        assert result["away_score"] is None
        assert result["home_shots"] == 0


class TestNormalizeStanding:
    def test_normalizes_standing(self):
        raw = {
            "strTeam": "Bayern Munich",
            "idTeam": "100",
            "intRank": "1",
            "intPlayed": "20",
            "intWin": "15",
            "intDraw": "3",
            "intLoss": "2",
            "intGoalsFor": "50",
            "intGoalsAgainst": "15",
            "intGoalDifference": "35",
            "intPoints": "48",
            "strForm": "WWDWW",
        }
        result = SportsDBFetcher._normalize_standing(raw)
        assert result["team"] == "Bayern Munich"
        assert result["position"] == 1
        assert result["won"] == 15
        assert result["goal_diff"] == 35
        assert result["form"] == "WWDWW"
        assert result["source"] == "thesportsdb"


class TestGetPastEvents:
    @patch("src.integrations.sportsdb_fetcher.cache")
    @patch("src.integrations.sportsdb_fetcher._safe_sync_run")
    def test_returns_normalized_events(self, mock_run, mock_cache):
        mock_cache.get_json.return_value = None
        mock_run.return_value = {
            "events": [
                {
                    "idEvent": "1",
                    "strHomeTeam": "A",
                    "strAwayTeam": "B",
                    "intHomeScore": "1",
                    "intAwayScore": "0",
                }
            ]
        }
        fetcher = SportsDBFetcher()
        events = fetcher.get_past_events("soccer_epl", rounds=5)
        assert len(events) == 1
        assert events[0]["home_team"] == "A"
        assert events[0]["source"] == "thesportsdb"

    @patch("src.integrations.sportsdb_fetcher.cache")
    def test_returns_empty_for_unknown_sport(self, mock_cache):
        fetcher = SportsDBFetcher()
        events = fetcher.get_past_events("unknown_sport")
        assert events == []

    @patch("src.integrations.sportsdb_fetcher.cache")
    def test_uses_cache(self, mock_cache):
        mock_cache.get_json.return_value = {"events": [{"idEvent": "cached"}]}
        fetcher = SportsDBFetcher()
        events = fetcher.get_past_events("soccer_epl")
        assert len(events) == 1


class TestLeagueTable:
    @patch("src.integrations.sportsdb_fetcher.cache")
    @patch("src.integrations.sportsdb_fetcher._safe_sync_run")
    def test_returns_standings(self, mock_run, mock_cache):
        mock_cache.get_json.return_value = None
        mock_run.return_value = {
            "table": [
                {"strTeam": "Bayern", "intRank": "1", "intPlayed": "20",
                 "intWin": "15", "intDraw": "3", "intLoss": "2",
                 "intGoalsFor": "50", "intGoalsAgainst": "15",
                 "intGoalDifference": "35", "intPoints": "48"}
            ]
        }
        fetcher = SportsDBFetcher()
        table = fetcher.get_league_table("soccer_germany_bundesliga")
        assert len(table) == 1
        assert table[0]["team"] == "Bayern"
