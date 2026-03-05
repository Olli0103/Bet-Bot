from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from scipy.stats import multivariate_normal, norm

from src.core.betting_math import (
    combo_odds,
    combo_probability,
    expected_value,
    kelly_fraction,
    kelly_stake,
)
from src.core.risk_guards import apply_stake_cap, check_data_source_health, explain_signal, get_dynamic_kelly_frac, passes_confidence_gate
from src.core.settings import settings
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
        model_probability_raw: float = 0.0,
        calibration_source: str = "",
        sharp_odds: float = 0.0,
        vig: float = 0.0,
    ) -> BetSignal:
        # Dynamic Kelly: shrink fraction based on model confidence (Brier score)
        effective_kelly_frac = get_dynamic_kelly_frac(sport, base_frac=kelly_frac)
        ev = expected_value(model_probability, bookmaker_odds, tax_rate=tax_rate)
        kf_raw = kelly_fraction(model_probability, bookmaker_odds, frac=effective_kelly_frac, tax_rate=tax_rate)
        stake_raw = round(kelly_stake(self.bankroll, kf_raw), 2)

        # --- Data source health gate ---
        source_ok, source_reason = check_data_source_health()
        rejected_reason = ""
        if not source_ok:
            rejected_reason = f"data_source_offline: {source_reason}"
            log.warning("Data source gate blocked signal: %s %s — %s",
                        sport, selection, rejected_reason)
            eff_raw_early = model_probability_raw if model_probability_raw > 0 else model_probability
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
                model_probability_raw=eff_raw_early,
                model_probability_calibrated=model_probability,
                calibration_source=calibration_source,
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

        # Effective raw prob: caller may pass 0.0 meaning "same as model_probability"
        eff_raw = model_probability_raw if model_probability_raw > 0 else model_probability

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
            model_probability_raw=eff_raw,
            model_probability_calibrated=model_probability,
            calibration_source=calibration_source,
        )

    @staticmethod
    def _build_correlation_matrix(combo_legs: List[ComboLeg]) -> np.ndarray:
        """Build a pairwise correlation matrix from the CorrelationEngine.

        Converts the pairwise multipliers from CorrelationEngine into
        Pearson correlation coefficients for use in the Gaussian copula.

        Multiplier → correlation mapping (empirical):
          multiplier 1.0  → rho 0.0  (independent)
          multiplier 1.15 → rho +0.25 (positive correlation)
          multiplier 0.70 → rho -0.40 (negative correlation)
          multiplier 0.92 → rho -0.10 (mild negative)

        Linear transform: rho ≈ clamp((mult - 1.0) * 1.5, -0.60, +0.60)
        """
        from src.core.correlation import CorrelationEngine

        n = len(combo_legs)
        corr = np.eye(n)

        legs_as_dicts = [
            {
                "event_id": leg.event_id,
                "selection": leg.selection,
                "odds": leg.odds,
                "probability": leg.probability,
                "sport": leg.sport,
                "market_type": leg.market_type,
                "market": leg.market,
                "home_team": leg.home_team,
                "away_team": leg.away_team,
            }
            for leg in combo_legs
        ]

        for i in range(n):
            for j in range(i + 1, n):
                mult = CorrelationEngine._pair_penalty(legs_as_dicts[i], legs_as_dicts[j])
                # Convert multiplier to correlation coefficient
                rho = max(-0.60, min(0.60, (mult - 1.0) * 1.5))
                corr[i, j] = rho
                corr[j, i] = rho

        return corr

    @staticmethod
    def _compute_joint_probability_copula(
        combo_legs: List[ComboLeg],
        correlation_matrix: np.ndarray,
    ) -> float:
        """Compute joint probability using a Gaussian copula.

        Maps each leg's marginal probability through the inverse normal
        CDF (quantile function) to get the copula space, then computes
        the joint CDF of the multivariate normal distribution.

        This properly handles correlated outcomes without breaking
        probability bounds — unlike the linear scalar multiplication
        which can produce p > 1.0.
        """
        n = len(combo_legs)
        if n == 1:
            return combo_legs[0].probability

        # Map marginal probabilities to normal quantiles
        quantiles = []
        for leg in combo_legs:
            p = max(1e-5, min(1.0 - 1e-5, leg.probability))
            quantiles.append(norm.ppf(p))

        try:
            joint_prob = multivariate_normal.cdf(
                quantiles,
                mean=np.zeros(n),
                cov=correlation_matrix,
            )
            result = float(max(1e-6, min(1.0 - 1e-6, joint_prob)))
        except Exception as exc:
            log.warning("Gaussian copula failed (%s), falling back to independent", exc)
            result = float(np.prod([leg.probability for leg in combo_legs]))

        return result

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
        # Gaussian copula: proper joint probability that respects [0, 1] bounds
        corr_matrix = self._build_correlation_matrix(combo_legs)
        p_copula = self._compute_joint_probability_copula(combo_legs, corr_matrix)

        # Dynamic Kelly for combos: use first leg's sport for model lookup
        combo_sport = combo_legs[0].sport if combo_legs else ""
        effective_kelly_frac = get_dynamic_kelly_frac(combo_sport, base_frac=kelly_frac)
        ev = expected_value(p_copula, odds, tax_rate=tax_rate)
        kf = kelly_fraction(p_copula, odds, frac=effective_kelly_frac, tax_rate=tax_rate)
        stake = round(kelly_stake(self.bankroll, kf), 2)

        # For backward compatibility, store the ratio vs independent as "correlation_penalty"
        p_independent = combo_probability(l.probability for l in combo_legs)
        effective_penalty = p_copula / max(1e-9, p_independent)

        return ComboBet(
            legs=combo_legs,
            combined_odds=odds,
            combined_probability=p_copula,
            correlation_penalty=round(effective_penalty, 4),
            expected_value=ev,
            kelly_fraction=kf,
            recommended_stake=stake,
        )

    def rank_value_bets(self, signals: List[BetSignal], min_ev: float = 0.0) -> List[BetSignal]:
        valid = [s for s in signals if s.expected_value > min_ev]
        return sorted(valid, key=lambda s: s.expected_value, reverse=True)
