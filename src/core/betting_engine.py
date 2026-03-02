from __future__ import annotations

from typing import Dict, List

from src.core.betting_math import (
    combo_odds,
    combo_probability,
    expected_value,
    kelly_fraction,
    kelly_stake,
)
from src.models.betting import BetSignal, ComboBet, ComboLeg


class BettingEngine:
    def __init__(self, bankroll: float):
        self.bankroll = bankroll

    def make_signal(
        self,
        sport: str,
        event_id: str,
        market: str,
        selection: str,
        bookmaker_odds: float,
        model_probability: float,
        kelly_frac: float = 0.2,
        source_mode: str = "primary",
        reference_book: str = "pinnacle",
        confidence: float = 1.0,
        tax_rate: float = 0.0,
    ) -> BetSignal:
        ev = expected_value(model_probability, bookmaker_odds, tax_rate=tax_rate)
        kf = kelly_fraction(model_probability, bookmaker_odds, frac=kelly_frac, tax_rate=tax_rate)
        stake = round(kelly_stake(self.bankroll, kf), 2)
        return BetSignal(
            sport=sport,
            event_id=event_id,
            market=market,
            selection=selection,
            bookmaker_odds=bookmaker_odds,
            model_probability=model_probability,
            expected_value=ev,
            kelly_fraction=kf,
            recommended_stake=stake,
            source_mode=source_mode,
            reference_book=reference_book,
            confidence=confidence,
        )

    def build_combo(
        self,
        legs: List[Dict],
        correlation_penalty: float = 0.9,
        kelly_frac: float = 0.1,
    ) -> ComboBet:
        combo_legs = [
            ComboLeg(
                event_id=l["event_id"],
                selection=l["selection"],
                odds=float(l["odds"]),
                probability=float(l["probability"]),
            )
            for l in legs
        ]

        odds = combo_odds(l.odds for l in combo_legs)
        p_independent = combo_probability(l.probability for l in combo_legs)
        p_adjusted = p_independent * correlation_penalty

        ev = expected_value(p_adjusted, odds)
        kf = kelly_fraction(p_adjusted, odds, frac=kelly_frac)
        stake = round(kelly_stake(self.bankroll, kf), 2)

        return ComboBet(
            legs=combo_legs,
            combined_odds=odds,
            combined_probability=p_adjusted,
            correlation_penalty=correlation_penalty,
            expected_value=ev,
            kelly_fraction=kf,
            recommended_stake=stake,
        )

    def rank_value_bets(self, signals: List[BetSignal], min_ev: float = 0.0) -> List[BetSignal]:
        valid = [s for s in signals if s.expected_value > min_ev]
        return sorted(valid, key=lambda s: s.expected_value, reverse=True)
