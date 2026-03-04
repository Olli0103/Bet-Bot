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
from src.core.risk_guards import apply_stake_cap, check_data_source_health, explain_signal, passes_confidence_gate
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

        # --- Data source health gate ---
        source_ok, source_reason = check_data_source_health()
        rejected_reason = ""
        if not source_ok:
            rejected_reason = f"data_source_offline: {source_reason}"
            log.warning("Data source gate blocked signal: %s %s — %s",
                        sport, selection, rejected_reason)
            return BetSignal(
                sport=sport, event_id=event_id, market=market,
                selection=selection, bookmaker_odds=bookmaker_odds,
                model_probability=model_probability, expected_value=ev,
                kelly_fraction=0.0, recommended_stake=0.0,
                source_mode=source_mode, reference_book=reference_book,
                source_quality=source_quality, confidence=model_probability,
                kelly_raw=kf_raw, stake_before_cap=stake_raw,
                stake_cap_applied=False, trigger=trigger,
                rejected_reason=rejected_reason, explanation="",
            )

        # --- Confidence gate (uses model_probability as THE confidence) ---
        passed, min_conf = passes_confidence_gate(model_probability, sport, market)
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

        # Generate natural-language explanation for the Telegram UI
        why = ""
        if not rejected_reason:
            try:
                why = explain_signal(
                    model_probability=model_probability,
                    expected_value=ev,
                    bookmaker_odds=bookmaker_odds,
                    sport=sport,
                )
            except Exception:
                pass

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
            explanation=why,
        )

    @staticmethod
    def _compute_correlation_penalty(combo_legs: List[ComboLeg]) -> float:
        """Compute a correlation penalty based on intra-event leg overlap.

        Legs from *different* events are assumed independent (penalty = 1.0).
        Legs sharing the same ``event_id`` (same-game parlay) are heavily
        correlated — multiplying independent probabilities drastically
        overestimates the combined EV.  A per-pair penalty of 0.80 is
        applied multiplicatively to compensate.
        """
        from collections import Counter
        event_counts = Counter(leg.event_id for leg in combo_legs)
        # Number of intra-event pairs that share a match
        correlated_pairs = sum(
            n * (n - 1) // 2 for n in event_counts.values() if n > 1
        )
        if correlated_pairs == 0:
            return 1.0
        # Warn on same-game parlays — they should normally be blocked upstream
        for eid, cnt in event_counts.items():
            if cnt > 1:
                log.warning(
                    "Same-game parlay detected: %d legs share event_id=%s "
                    "(correlation penalty applied)", cnt, eid,
                )
        # 0.80 per correlated pair (stronger than previous 0.90 to reflect
        # high intra-match correlation, e.g. "Home Win" + "Over 2.5")
        return 0.80 ** correlated_pairs

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
        # Use event-aware correlation instead of flat penalty
        effective_penalty = self._compute_correlation_penalty(combo_legs)
        p_adjusted = p_independent * effective_penalty

        ev = expected_value(p_adjusted, odds, tax_rate=tax_rate)
        kf = kelly_fraction(p_adjusted, odds, frac=kelly_frac, tax_rate=tax_rate)
        stake = round(kelly_stake(self.bankroll, kf), 2)

        return ComboBet(
            legs=combo_legs,
            combined_odds=odds,
            combined_probability=p_adjusted,
            correlation_penalty=effective_penalty,
            expected_value=ev,
            kelly_fraction=kf,
            recommended_stake=stake,
        )

    def rank_value_bets(self, signals: List[BetSignal], min_ev: float = 0.0) -> List[BetSignal]:
        valid = [s for s in signals if s.expected_value > min_ev]
        return sorted(valid, key=lambda s: s.expected_value, reverse=True)
