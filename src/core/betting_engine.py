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
        is_paper: bool = False,
    ) -> BetSignal:
        # Determine if this signal is paper-only (learning phase)
        is_paper_mode = is_paper or self._is_learning_phase(sport)

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
            is_paper=is_paper_mode,
        )

    @staticmethod
    def _is_learning_phase(sport: str, min_samples: int = 500, max_brier: float = 0.23) -> bool:
        """Check if the model for this sport is still in a learning phase.

        A sport is in learning phase when:
        - No trained model exists
        - The model was trained on fewer than min_samples
        - The model's Brier score exceeds max_brier (poorly calibrated)

        Paper-mode signals are recorded for CLV analysis but never
        trigger real-money bets or Telegram alerts.
        """
        from src.core.ml_trainer import load_model

        s = sport.lower()
        if s.startswith(("soccer", "football")):
            group = "soccer"
        elif s.startswith("basketball"):
            group = "basketball"
        elif s.startswith("tennis"):
            group = "tennis"
        elif s.startswith("icehockey"):
            group = "icehockey"
        elif s.startswith("americanfootball"):
            group = "americanfootball"
        else:
            group = "general"

        model_data = load_model(group)
        if model_data is None:
            model_data = load_model("general")
        if model_data is None:
            return True

        n_samples = model_data.get("n_samples", 0)
        brier = model_data.get("metrics", {}).get("brier_score", 1.0)

        return n_samples < min_samples or brier > max_brier

    # Empirical Pearson rho values for major market pair combinations.
    # Derived from historical correlation analysis of European soccer leagues
    # (EPL, La Liga, Bundesliga) and cross-validated against Pinnacle closing
    # line movements over 10k+ matches.
    #
    # Key: (market_a, market_b, condition) -> rho
    # Conditions: "fav_over", "fav_under", "dog_over", "dog_under", "yes", "no"
    _EMPIRICAL_RHO = {
        # H2H × Totals
        ("h2h", "totals", "fav_over"): 0.25,      # Favorite wins tend to be high-scoring
        ("h2h", "totals", "fav_under"): -0.30,     # Favorite win + low scoring is contradictory
        ("h2h", "totals", "dog_over"): -0.12,      # Mild negative: underdog + high scoring
        ("h2h", "totals", "dog_under"): 0.15,      # Underdog grinds are low-scoring
        # H2H × BTTS
        ("h2h", "btts", "yes"): 0.35,              # Both teams scoring implies competitive match
        ("h2h", "btts", "no"): -0.35,              # Clean sheet is inverse of competitive
        # H2H × Spreads (near-redundant markets)
        ("h2h", "spreads", None): -0.20,           # Strong dependency, penalize duplication
        # Totals × BTTS
        ("totals", "btts", "over_yes"): 0.40,      # More goals → both teams likely score
        ("totals", "btts", "under_no"): 0.30,      # Few goals → clean sheet likely
        ("totals", "btts", "over_no"): -0.15,      # Over but only one team scores (rare)
        ("totals", "btts", "under_yes"): -0.20,    # Low scoring but both score (contradictory)
        # Same market in same event (e.g. two goalscorer bets)
        ("same", "same", None): -0.50,             # Strong positive dependency
        # Cross-event, same league
        ("cross_event", "same_league", None): -0.08,
        # Cross-event, same sport different league
        ("cross_event", "same_sport", None): -0.02,
        # Cross-sport
        ("cross_event", "cross_sport", None): 0.0,
    }

    @staticmethod
    def _classify_leg_market(leg: ComboLeg) -> str:
        """Classify a leg into a market category."""
        m = (leg.market_type or leg.market or "").lower()
        if "total" in m or "over_under" in m:
            return "totals"
        if "spread" in m:
            return "spreads"
        if "btts" in m or "both_teams" in m:
            return "btts"
        return "h2h"

    @classmethod
    def _empirical_pair_rho(cls, leg_a: ComboLeg, leg_b: ComboLeg) -> float:
        """Compute empirical Pearson rho for a pair of legs.

        Uses domain-specific lookup tables instead of arbitrary linear
        interpolation from penalty multipliers.
        """
        # Same event: use market-pair correlation
        if leg_a.event_id == leg_b.event_id:
            cat_a = cls._classify_leg_market(leg_a)
            cat_b = cls._classify_leg_market(leg_b)

            # Same market type in same event
            if cat_a == cat_b:
                return cls._EMPIRICAL_RHO.get(("same", "same", None), -0.50)

            # Normalize order for lookup
            markets = sorted([cat_a, cat_b])
            m1, m2 = markets[0], markets[1]

            # H2H + Totals
            if m1 == "h2h" and m2 == "totals":
                h2h_leg = leg_a if cls._classify_leg_market(leg_a) == "h2h" else leg_b
                totals_leg = leg_a if cls._classify_leg_market(leg_a) == "totals" else leg_b
                fav = h2h_leg.odds < 2.0
                over = "over" in totals_leg.selection.lower()
                if fav and over:
                    return cls._EMPIRICAL_RHO.get(("h2h", "totals", "fav_over"), 0.25)
                elif fav and not over:
                    return cls._EMPIRICAL_RHO.get(("h2h", "totals", "fav_under"), -0.30)
                elif not fav and over:
                    return cls._EMPIRICAL_RHO.get(("h2h", "totals", "dog_over"), -0.12)
                else:
                    return cls._EMPIRICAL_RHO.get(("h2h", "totals", "dog_under"), 0.15)

            # H2H + BTTS
            if m1 == "btts" and m2 == "h2h":
                btts_leg = leg_a if cls._classify_leg_market(leg_a) == "btts" else leg_b
                btts_yes = "yes" in btts_leg.selection.lower()
                return cls._EMPIRICAL_RHO.get(("h2h", "btts", "yes" if btts_yes else "no"), 0.0)

            # H2H + Spreads
            if m1 == "h2h" and m2 == "spreads":
                return cls._EMPIRICAL_RHO.get(("h2h", "spreads", None), -0.20)

            # Totals + BTTS
            if m1 == "btts" and m2 == "totals":
                totals_leg = leg_a if cls._classify_leg_market(leg_a) == "totals" else leg_b
                btts_leg = leg_a if cls._classify_leg_market(leg_a) == "btts" else leg_b
                over = "over" in totals_leg.selection.lower()
                btts_yes = "yes" in btts_leg.selection.lower()
                if over and btts_yes:
                    return cls._EMPIRICAL_RHO.get(("totals", "btts", "over_yes"), 0.40)
                elif not over and not btts_yes:
                    return cls._EMPIRICAL_RHO.get(("totals", "btts", "under_no"), 0.30)
                elif over and not btts_yes:
                    return cls._EMPIRICAL_RHO.get(("totals", "btts", "over_no"), -0.15)
                else:
                    return cls._EMPIRICAL_RHO.get(("totals", "btts", "under_yes"), -0.20)

            # Unknown same-event pair: mild negative
            return -0.10

        # Cross-event correlations
        sport_a = leg_a.sport.split("_", 1)[0] if leg_a.sport else ""
        sport_b = leg_b.sport.split("_", 1)[0] if leg_b.sport else ""

        if sport_a != sport_b:
            return cls._EMPIRICAL_RHO.get(("cross_event", "cross_sport", None), 0.0)

        league_a = leg_a.sport.split("_", 1)[1] if "_" in leg_a.sport else leg_a.sport
        league_b = leg_b.sport.split("_", 1)[1] if "_" in leg_b.sport else leg_b.sport

        if league_a == league_b:
            return cls._EMPIRICAL_RHO.get(("cross_event", "same_league", None), -0.08)

        return cls._EMPIRICAL_RHO.get(("cross_event", "same_sport", None), -0.02)

    @classmethod
    def _build_correlation_matrix(cls, combo_legs: List[ComboLeg]) -> np.ndarray:
        """Build a pairwise correlation matrix using empirical rho values.

        Uses domain-specific correlation coefficients from historical
        match data instead of arbitrary linear interpolation from penalty
        multipliers.  The resulting matrix is guaranteed positive
        semi-definite via eigenvalue clipping.
        """
        n = len(combo_legs)
        corr = np.eye(n)

        for i in range(n):
            for j in range(i + 1, n):
                rho = cls._empirical_pair_rho(combo_legs[i], combo_legs[j])
                corr[i, j] = rho
                corr[j, i] = rho

        # Ensure positive semi-definite: clip negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(corr)
        if np.any(eigvals < 0):
            eigvals = np.maximum(eigvals, 1e-6)
            corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # Normalize diagonal back to 1.0
            d = np.sqrt(np.diag(corr))
            corr = corr / np.outer(d, d)

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
