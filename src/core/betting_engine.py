from __future__ import annotations

import logging
from typing import Dict, List

from src.core.betting_math import (
    combo_odds,
    combo_probability,
    expected_value,
    kelly_fraction,
    kelly_stake,
)
from src.core.risk_guards import apply_stake_cap, passes_confidence_gate
from src.models.betting import BetSignal, ComboBet, ComboLeg

log = logging.getLogger(__name__)


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
        source_quality: float = 1.0,
        tax_rate: float = 0.0,
        trigger: str = "",
    ) -> BetSignal:
        ev = expected_value(model_probability, bookmaker_odds, tax_rate=tax_rate)
        kf_raw = kelly_fraction(model_probability, bookmaker_odds, frac=kelly_frac, tax_rate=tax_rate)
        stake_raw = round(kelly_stake(self.bankroll, kf_raw), 2)

        # --- Confidence gate (uses model_probability as THE confidence) ---
        passed, min_conf = passes_confidence_gate(model_probability, sport, market)
        rejected_reason = ""
        if not passed:
            rejected_reason = (
                f"reject_confidence_below_min: "
                f"model_prob={model_probability:.2f} < gate={min_conf:.2f}"
            )
            log.info("Confidence gate blocked: %s %s %s — %s",
                     sport, event_id, selection, rejected_reason)
            # Zero out stake so this signal is filtered out as non-playable
            stake_final = 0.0
            kf = 0.0
        else:
            kf = kf_raw
            # --- Stake cap ---
            stake_final, was_capped = apply_stake_cap(
                stake_raw, self.bankroll, bookmaker_odds, market, selection,
            )
            if was_capped:
                log.debug("Stake capped: %s %s %.2f -> %.2f",
                          sport, selection, stake_raw, stake_final)
            log.info(
                "Signal accepted: %s %s model_prob=%.2f ev=%.4f trigger=%s stake=%.2f",
                sport, selection, model_probability, ev, trigger or "none", stake_final,
            )

        return BetSignal(
            sport=sport,
            event_id=event_id,
            market=market,
            selection=selection,
            bookmaker_odds=bookmaker_odds,
            model_probability=model_probability,
            expected_value=ev,
            kelly_fraction=kf,
            recommended_stake=stake_final,
            source_mode=source_mode,
            reference_book=reference_book,
            source_quality=source_quality,
            # confidence == model_probability: single source of truth for UI + ranking
            confidence=model_probability,
            kelly_raw=kf_raw,
            stake_before_cap=stake_raw,
            stake_cap_applied=stake_final < stake_raw and stake_final > 0,
            trigger=trigger,
            rejected_reason=rejected_reason,
        )

    def build_combo(
        self,
        legs: List[Dict],
        correlation_penalty: float = 0.9,
        kelly_frac: float = 0.1,
        tax_rate: float = 0.0,
    ) -> ComboBet:
        combo_legs = [
            ComboLeg(
                event_id=l["event_id"],
                selection=l["selection"],
                odds=float(l["odds"]),
                probability=float(l["probability"]),
                sport=l.get("sport", ""),
                market_type=l.get("market_type", "h2h"),
                home_team=l.get("home_team", ""),
                away_team=l.get("away_team", ""),
                market=l.get("market", l.get("market_type", "h2h")),
            )
            for l in legs
        ]

        odds = combo_odds(l.odds for l in combo_legs)
        p_independent = combo_probability(l.probability for l in combo_legs)
        p_adjusted = p_independent * correlation_penalty

        ev = expected_value(p_adjusted, odds, tax_rate=tax_rate)
        kf = kelly_fraction(p_adjusted, odds, frac=kelly_frac, tax_rate=tax_rate)
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
