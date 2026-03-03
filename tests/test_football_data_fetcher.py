"""Tests for src/integrations/football_data_fetcher.py — football-data.org integration."""
from unittest.mock import patch, MagicMock

import pytest

from src.integrations.football_data_fetcher import FootballDataFetcher


class TestNormalizeMatch:
    def test_normalizes_complete_match(self):
        raw = {
            "id": 330299,
            "utcDate": "2026-02-27T16:05:00Z",
            "status": "FINISHED",
            "matchday": 26,
            "homeTeam": {
                "id": 531,
                "name": "FC Bayern Munich",
                "shortName": "Bayern",
            },
            "awayTeam": {
                "id": 524,
                "name": "Borussia Dortmund",
                "shortName": "Dortmund",
            },
            "score": {
                "winner": "HOME_TEAM",
                "fullTime": {"home": 2, "away": 1},
                "halfTime": {"home": 1, "away": 0},
            },
        }
        result = FootballDataFetcher._normalize_match(raw)
        assert result["match_id"] == 330299
        assert result["home_team"] == "FC Bayern Munich"
        assert result["away_team"] == "Borussia Dortmund"
        assert result["home_score"] == 2
        assert result["away_score"] == 1
        assert result["home_ht_score"] == 1
        assert result["away_ht_score"] == 0
        assert result["winner"] == "HOME_TEAM"
        assert result["source"] == "football-data.org"

    def test_normalizes_match_no_score(self):
        raw = {
            "id": 999,
            "utcDate": "2026-03-05T20:00:00Z",
            "status": "SCHEDULED",
            "matchday": 27,
            "homeTeam": {"id": 1, "name": "TeamA"},
            "awayTeam": {"id": 2, "name": "TeamB"},
            "score": {},
        }
        result = FootballDataFetcher._normalize_match(raw)
        assert result["home_score"] is None
        assert result["away_score"] is None


class TestNormalizeStanding:
    def test_normalizes_standing(self):
        raw = {
            "position": 1,
            "team": {
                "id": 86,
                "name": "Real Madrid CF",
                "shortName": "Real Madrid",
            },
            "playedGames": 34,
            "form": "W,W,W,W,W",
            "won": 25,
            "draw": 6,
            "lost": 3,
            "points": 81,
            "goalsFor": 73,
            "goalsAgainst": 29,
            "goalDifference": 44,
        }
        result = FootballDataFetcher._normalize_standing(raw)
        assert result["team"] == "Real Madrid CF"
        assert result["position"] == 1
        assert result["won"] == 25
        assert result["goal_diff"] == 44
        assert result["source"] == "football-data.org"


class TestGetMatches:
    @patch("src.integrations.football_data_fetcher.cache")
    @patch("src.integrations.football_data_fetcher._safe_sync_run")
    def test_returns_matches_for_known_league(self, mock_run, mock_cache):
        mock_cache.get_json.return_value = None
        mock_run.return_value = {
            "matches": [
                {
                    "id": 1,
                    "utcDate": "2026-03-01T15:00:00Z",
                    "status": "FINISHED",
                    "matchday": 23,
                    "homeTeam": {"id": 1, "name": "Home"},
                    "awayTeam": {"id": 2, "name": "Away"},
                    "score": {"fullTime": {"home": 1, "away": 0}, "halfTime": {}},
                }
            ]
        }
        fetcher = FootballDataFetcher(api_key="test_key")
        matches = fetcher.get_matches("soccer_epl", status="FINISHED")
        assert len(matches) == 1
        assert matches[0]["home_team"] == "Home"

    @patch("src.integrations.football_data_fetcher.cache")
    def test_returns_empty_for_unknown_league(self, mock_cache):
        fetcher = FootballDataFetcher()
        matches = fetcher.get_matches("basketball_nba")
        assert matches == []


class TestGetStandings:
    @patch("src.integrations.football_data_fetcher.cache")
    @patch("src.integrations.football_data_fetcher._safe_sync_run")
    def test_returns_standings(self, mock_run, mock_cache):
        mock_cache.get_json.return_value = None
        mock_run.return_value = {
            "standings": [
                {
                    "type": "TOTAL",
                    "table": [
                        {
                            "position": 1,
                            "team": {"id": 1, "name": "TopTeam"},
                            "playedGames": 20,
                            "won": 15,
                            "draw": 3,
                            "lost": 2,
                            "points": 48,
                            "goalsFor": 45,
                            "goalsAgainst": 12,
                            "goalDifference": 33,
                        }
                    ],
                }
            ]
        }
        fetcher = FootballDataFetcher(api_key="test_key")
        standings = fetcher.get_standings("soccer_germany_bundesliga")
        assert len(standings) == 1
        assert standings[0]["team"] == "TopTeam"
