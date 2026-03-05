"""Gaussian Copula-based correlation engine for combo bets.

Replaces legacy scalar multipliers with a proper bivariate/multivariate
Gaussian copula that maps marginal probabilities through the inverse
normal CDF and computes joint probability via the multivariate normal CDF.

This ensures:
  - Joint probabilities always remain in [0, 1]
  - Positive correlations correctly *boost* joint probability
  - Negative correlations correctly *reduce* joint probability
  - The correlation matrix is guaranteed positive semi-definite

The empirical rho values are domain-specific Pearson correlations derived
from historical match data (same lookup table as betting_engine.py).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm, multivariate_normal

log = logging.getLogger(__name__)


# ---- Empirical rho lookup table ------------------------------------------------
# Key: (market_a, market_b, condition) -> Pearson rho
# Derived from 10k+ European soccer matches' line movements.

_EMPIRICAL_RHO: Dict[Tuple[str, str, Optional[str]], float] = {
    # H2H x Totals
    ("h2h", "totals", "fav_over"): 0.25,
    ("h2h", "totals", "fav_under"): -0.30,
    ("h2h", "totals", "dog_over"): -0.12,
    ("h2h", "totals", "dog_under"): 0.15,
    # H2H x BTTS
    ("h2h", "btts", "yes"): 0.35,
    ("h2h", "btts", "no"): -0.35,
    # H2H x Spreads (near-redundant markets)
    ("h2h", "spreads", None): -0.20,
    # Totals x BTTS
    ("totals", "btts", "over_yes"): 0.40,
    ("totals", "btts", "under_no"): 0.30,
    ("totals", "btts", "over_no"): -0.15,
    ("totals", "btts", "under_yes"): -0.20,
    # Same market in same event (e.g. two goalscorer bets)
    ("same", "same", None): 0.50,
    # Cross-event correlations
    ("cross_event", "same_league", None): 0.08,
    ("cross_event", "same_sport", None): 0.02,
    ("cross_event", "cross_sport", None): 0.0,
}


# ---- sport & league extraction -----------------------------------------------

def _extract_league(sport_key: str) -> str:
    """Return the league portion of a sport key (e.g., 'soccer_epl' -> 'epl')."""
    parts = sport_key.split("_", 1)
    return parts[1] if len(parts) > 1 else sport_key


def _extract_sport(sport_key: str) -> str:
    """Return the sport prefix (e.g., 'soccer_epl' -> 'soccer')."""
    return sport_key.split("_", 1)[0]


# ---- market classification ----------------------------------------------------

def _classify_market(leg: Dict) -> str:
    """Classify a leg into a market category for correlation lookup."""
    market = leg.get("market", leg.get("market_type", "")).lower()
    if "total" in market or "over_under" in market:
        return "totals"
    if "spread" in market:
        return "spreads"
    if "btts" in market or "both_teams" in market:
        return "btts"
    return "h2h"


def _is_over(leg: Dict) -> bool:
    """Check if the selection is an 'over' bet."""
    return "over" in leg.get("selection", "").lower()


def _is_favorite(leg: Dict) -> bool:
    """Heuristic: odds < 2.0 implies favorite."""
    return leg.get("odds", 99.0) < 2.0


def _is_btts_yes(leg: Dict) -> bool:
    """Check if the selection is BTTS Yes."""
    return "yes" in leg.get("selection", "").lower()


# ---- pairwise rho computation -------------------------------------------------

def _empirical_pair_rho(leg_a: Dict, leg_b: Dict) -> float:
    """Compute empirical Pearson rho for a pair of legs.

    Returns a correlation coefficient in [-1, 1] for the Gaussian copula,
    using domain-specific lookup tables derived from historical match data.
    """
    event_a = leg_a.get("event_id", "")
    event_b = leg_b.get("event_id", "")

    # Same event: use market-pair correlation
    if event_a and event_b and event_a == event_b:
        cat_a = _classify_market(leg_a)
        cat_b = _classify_market(leg_b)

        # Same market type in same event
        if cat_a == cat_b:
            return _EMPIRICAL_RHO.get(("same", "same", None), 0.50)

        # Normalize order for consistent lookup
        markets = sorted([cat_a, cat_b])
        m1, m2 = markets[0], markets[1]

        # H2H + Totals
        if m1 == "h2h" and m2 == "totals":
            h2h_leg = leg_a if cat_a == "h2h" else leg_b
            totals_leg = leg_a if cat_a == "totals" else leg_b
            fav = _is_favorite(h2h_leg)
            over = _is_over(totals_leg)
            if fav and over:
                return _EMPIRICAL_RHO.get(("h2h", "totals", "fav_over"), 0.25)
            elif fav and not over:
                return _EMPIRICAL_RHO.get(("h2h", "totals", "fav_under"), -0.30)
            elif not fav and over:
                return _EMPIRICAL_RHO.get(("h2h", "totals", "dog_over"), -0.12)
            else:
                return _EMPIRICAL_RHO.get(("h2h", "totals", "dog_under"), 0.15)

        # H2H + BTTS
        if m1 == "btts" and m2 == "h2h":
            btts_leg = leg_a if cat_a == "btts" else leg_b
            btts_yes = _is_btts_yes(btts_leg)
            return _EMPIRICAL_RHO.get(("h2h", "btts", "yes" if btts_yes else "no"), 0.0)

        # H2H + Spreads
        if m1 == "h2h" and m2 == "spreads":
            return _EMPIRICAL_RHO.get(("h2h", "spreads", None), -0.20)

        # Totals + BTTS
        if m1 == "btts" and m2 == "totals":
            totals_leg = leg_a if cat_a == "totals" else leg_b
            btts_leg = leg_a if cat_a == "btts" else leg_b
            over = _is_over(totals_leg)
            btts_yes = _is_btts_yes(btts_leg)
            if over and btts_yes:
                return _EMPIRICAL_RHO.get(("totals", "btts", "over_yes"), 0.40)
            elif not over and not btts_yes:
                return _EMPIRICAL_RHO.get(("totals", "btts", "under_no"), 0.30)
            elif over and not btts_yes:
                return _EMPIRICAL_RHO.get(("totals", "btts", "over_no"), -0.15)
            else:
                return _EMPIRICAL_RHO.get(("totals", "btts", "under_yes"), -0.20)

        # Unknown same-event pair
        return 0.10

    # Cross-event correlations
    sport_a = _extract_sport(leg_a.get("sport", ""))
    sport_b = _extract_sport(leg_b.get("sport", ""))

    if sport_a != sport_b:
        return _EMPIRICAL_RHO.get(("cross_event", "cross_sport", None), 0.0)

    league_a = _extract_league(leg_a.get("sport", ""))
    league_b = _extract_league(leg_b.get("sport", ""))

    if league_a == league_b:
        return _EMPIRICAL_RHO.get(("cross_event", "same_league", None), 0.08)

    return _EMPIRICAL_RHO.get(("cross_event", "same_sport", None), 0.02)


def _build_correlation_matrix(legs: List[Dict]) -> np.ndarray:
    """Build a pairwise correlation matrix using empirical rho values.

    Guaranteed positive semi-definite via eigenvalue clipping.
    """
    n = len(legs)
    corr = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            rho = _empirical_pair_rho(legs[i], legs[j])
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


def bivariate_copula_probability(p1: float, p2: float, rho: float) -> float:
    """Compute joint probability of two events using a bivariate Gaussian copula.

    Parameters
    ----------
    p1, p2 : float
        Marginal probabilities of each leg.
    rho : float
        Pearson correlation coefficient between the two outcomes.

    Returns
    -------
    float
        Joint probability in (0, 1).
    """
    p1 = max(1e-5, min(1.0 - 1e-5, p1))
    p2 = max(1e-5, min(1.0 - 1e-5, p2))
    rho = max(-0.99, min(0.99, rho))

    z1 = norm.ppf(p1)
    z2 = norm.ppf(p2)

    try:
        cov = np.array([[1.0, rho], [rho, 1.0]])
        joint = multivariate_normal.cdf([z1, z2], mean=[0, 0], cov=cov)
        return float(max(1e-6, min(1.0 - 1e-6, joint)))
    except Exception as exc:
        log.warning("Bivariate copula failed (%s), falling back to independent", exc)
        return p1 * p2


# ---- main engine --------------------------------------------------------------

class CorrelationEngine:
    """Computes joint probability adjustments using Gaussian copulas.

    Replaces the legacy scalar-multiplier approach with proper multivariate
    normal CDF computation that maps marginal probabilities through the
    inverse normal CDF (quantile function) into copula space.
    """

    @staticmethod
    def _pair_rho(leg_a: Dict, leg_b: Dict) -> float:
        """Return the empirical Pearson rho for a pair of legs."""
        return _empirical_pair_rho(leg_a, leg_b)

    @classmethod
    def compute_joint_probability(cls, legs: List[Dict]) -> float:
        """Compute joint probability using a multivariate Gaussian copula.

        Maps each leg's marginal probability through the inverse normal
        CDF, then computes the joint CDF of the multivariate normal.

        Parameters
        ----------
        legs : list of dict
            Each leg must have 'probability', 'event_id', 'sport',
            'market'/'market_type', 'selection', and 'odds' keys.

        Returns
        -------
        float
            Joint probability respecting correlation structure.
        """
        n = len(legs)
        if n == 0:
            return 0.0
        if n == 1:
            return float(legs[0].get("probability", 0.5))

        # Build correlation matrix
        corr_matrix = _build_correlation_matrix(legs)

        # Map marginal probabilities to normal quantiles
        quantiles = []
        for leg in legs:
            p = max(1e-5, min(1.0 - 1e-5, float(leg.get("probability", 0.5))))
            quantiles.append(norm.ppf(p))

        try:
            joint_prob = multivariate_normal.cdf(
                quantiles,
                mean=np.zeros(n),
                cov=corr_matrix,
            )
            result = float(max(1e-6, min(1.0 - 1e-6, joint_prob)))
        except Exception as exc:
            log.warning("Gaussian copula failed (%s), falling back to independent", exc)
            result = float(np.prod([
                float(leg.get("probability", 0.5)) for leg in legs
            ]))

        log.debug(
            "Copula joint prob: %.6f (n=%d, independent=%.6f)",
            result, n,
            float(np.prod([float(leg.get("probability", 0.5)) for leg in legs])),
        )
        return result

    @classmethod
    def compute_combo_correlation(cls, legs: List[Dict]) -> float:
        """Backward-compatible: return multiplier relative to independent product.

        This method preserves the interface for callers that expect a scalar
        multiplier.  Internally, it computes the copula joint probability and
        divides by the independent product to produce the adjustment factor.

        Final result is clamped to [0.50, 2.50] to prevent extreme values.
        """
        if len(legs) <= 1:
            return 1.0

        p_copula = cls.compute_joint_probability(legs)
        p_independent = float(np.prod([
            max(1e-9, float(leg.get("probability", 0.5))) for leg in legs
        ]))

        if p_independent < 1e-9:
            return 1.0

        multiplier = p_copula / p_independent
        return max(0.50, min(2.50, multiplier))
