"""Tests for src/models/betting.py — Pydantic data models."""
import pytest
from pydantic import ValidationError

from src.models.betting import BetSignal, ComboLeg, ComboBet


class TestBetSignal:
    def test_valid_signal(self):
        s = BetSignal(
            sport="basketball_nba",
            event_id="nba_001",
            market="h2h",
            selection="Lakers",
            bookmaker_odds=2.0,
            model_probability=0.55,
            expected_value=0.10,
            kelly_fraction=0.04,
            recommended_stake=5.0,
        )
        assert s.sport == "basketball_nba"
        assert s.model_probability == 0.55

    def test_odds_must_be_gt_1(self):
        with pytest.raises(ValidationError):
            BetSignal(
                sport="nba", event_id="1", market="h2h", selection="X",
                bookmaker_odds=1.0,  # must be > 1.0
                model_probability=0.5, expected_value=0.0,
                kelly_fraction=0.0, recommended_stake=0.0,
            )

    def test_probability_bounds(self):
        with pytest.raises(ValidationError):
            BetSignal(
                sport="nba", event_id="1", market="h2h", selection="X",
                bookmaker_odds=2.0,
                model_probability=0.0,  # must be > 0.0
                expected_value=0.0, kelly_fraction=0.0, recommended_stake=0.0,
            )
        with pytest.raises(ValidationError):
            BetSignal(
                sport="nba", event_id="1", market="h2h", selection="X",
                bookmaker_odds=2.0,
                model_probability=1.0,  # must be < 1.0
                expected_value=0.0, kelly_fraction=0.0, recommended_stake=0.0,
            )

    def test_defaults(self):
        s = BetSignal(
            sport="nba", event_id="1", market="h2h", selection="X",
            bookmaker_odds=2.0, model_probability=0.5,
            expected_value=0.0, kelly_fraction=0.0, recommended_stake=0.0,
        )
        assert s.source_mode == "primary"
        assert s.reference_book == "pinnacle"
        assert s.confidence == 1.0
        assert s.is_stale is False


class TestComboLeg:
    def test_valid_leg(self):
        leg = ComboLeg(
            event_id="ev1", selection="Home", odds=1.5, probability=0.65,
            sport="soccer_epl", home_team="Arsenal", away_team="Chelsea",
            market="h2h",
        )
        assert leg.home_team == "Arsenal"
        assert leg.market == "h2h"

    def test_defaults(self):
        leg = ComboLeg(event_id="ev1", selection="X", odds=2.0, probability=0.5)
        assert leg.sport == ""
        assert leg.home_team == ""
        assert leg.away_team == ""
        assert leg.market == ""
        assert leg.market_type == "h2h"


class TestComboBet:
    def test_valid_combo(self):
        legs = [
            ComboLeg(event_id="1", selection="A", odds=1.5, probability=0.65),
            ComboLeg(event_id="2", selection="B", odds=2.0, probability=0.50),
            ComboLeg(event_id="3", selection="C", odds=1.8, probability=0.55),
        ]
        combo = ComboBet(
            legs=legs,
            combined_odds=5.4,
            combined_probability=0.179,
            correlation_penalty=0.05,
            expected_value=0.50,
            kelly_fraction=0.01,
            recommended_stake=1.0,
        )
        assert len(combo.legs) == 3
        assert combo.combined_odds == 5.4
