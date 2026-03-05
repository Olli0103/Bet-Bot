"""Simultaneous Kelly Portfolio Sizing (Mean-Variance Optimization).

When multiple bets fire simultaneously (e.g. 5 Bundesliga games at 15:30),
sizing each independently with Kelly leads to catastrophic over-exposure
because correlated outcomes aren't accounted for.

This module implements the matrix form of Kelly:
    f* = Sigma^{-1} * mu
where:
    mu  = vector of expected edges (EV per bet)
    Sigma = covariance matrix of bet outcomes

The result is a set of optimal fractions that maximize the portfolio's
log-growth rate under the constraint of a maximum total exposure cap.

For uncorrelated bets, this reduces to independent Kelly (as expected).
For correlated bets (e.g. same-league matches, Over/Under + H2H on same
match), it automatically reduces exposure to avoid concentration risk.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)

# Maximum fraction of bankroll that can be risked across all simultaneous bets
MAX_TOTAL_EXPOSURE = 0.15  # 15% hard cap


def simultaneous_kelly_sizing(
    edges: np.ndarray,
    cov_matrix: np.ndarray,
    kelly_fraction: float = 0.25,
    max_total_exposure: float = MAX_TOTAL_EXPOSURE,
) -> np.ndarray:
    """Compute optimal bet fractions for a portfolio of simultaneous bets.

    Uses mean-variance approximation to Kelly:
        maximize  mu^T f - 0.5 f^T Sigma f
        subject to  sum(f) <= max_total_exposure, f >= 0

    Parameters
    ----------
    edges : np.ndarray
        Expected value (net EV) per bet, shape (n_bets,).
    cov_matrix : np.ndarray
        Covariance matrix of bet outcomes, shape (n_bets, n_bets).
        Diagonal = variance of each bet, off-diagonal = covariance.
    kelly_fraction : float
        Fractional Kelly multiplier (0.25 = quarter-Kelly).
    max_total_exposure : float
        Hard cap on sum of all fractions.

    Returns
    -------
    np.ndarray
        Optimal fractions, shape (n_bets,).  Zero for bets that should
        be skipped.
    """
    n_bets = len(edges)
    if n_bets == 0:
        return np.array([])

    # Validate inputs
    edges = np.asarray(edges, dtype=float)
    cov_matrix = np.asarray(cov_matrix, dtype=float)

    if cov_matrix.shape != (n_bets, n_bets):
        log.warning("Covariance matrix shape mismatch, falling back to independent Kelly")
        return _independent_kelly_fallback(edges, kelly_fraction, max_total_exposure)

    # Ensure PSD (add small ridge if needed)
    min_eig = np.linalg.eigvalsh(cov_matrix).min()
    if min_eig < 1e-8:
        cov_matrix = cov_matrix + np.eye(n_bets) * (1e-8 - min_eig)

    # Objective: maximize mu^T f - 0.5 f^T Sigma f  (minimize the negative)
    def objective(f):
        return -(np.dot(edges, f) - 0.5 * np.dot(f.T, np.dot(cov_matrix, f)))

    # Jacobian for faster convergence
    def jacobian(f):
        return -(edges - np.dot(cov_matrix, f))

    bounds = [(0.0, max_total_exposure) for _ in range(n_bets)]
    constraints = [{"type": "ineq", "fun": lambda f: max_total_exposure - np.sum(f)}]

    f0 = np.ones(n_bets) * (max_total_exposure / n_bets)

    result = minimize(
        objective, f0,
        method="SLSQP",
        jac=jacobian,
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        log.warning("Portfolio optimizer did not converge: %s", result.message)
        return _independent_kelly_fallback(edges, kelly_fraction, max_total_exposure)

    optimal = result.x * kelly_fraction
    # Zero out negligible fractions
    optimal[optimal < 1e-6] = 0.0
    return np.round(optimal, 6)


def _independent_kelly_fallback(
    edges: np.ndarray,
    kelly_fraction: float,
    max_total_exposure: float,
) -> np.ndarray:
    """Fallback: independent Kelly with proportional scaling to fit exposure cap."""
    raw = np.maximum(edges * kelly_fraction, 0.0)
    total = raw.sum()
    if total > max_total_exposure and total > 0:
        raw = raw * (max_total_exposure / total)
    return np.round(raw, 6)


def build_covariance_matrix(
    n_bets: int,
    event_ids: List[str],
    markets: List[str],
    sports: List[str],
    base_variance: float = 0.25,
) -> np.ndarray:
    """Build a covariance matrix from bet metadata.

    Heuristic correlation rules:
    - Same event, different market (H2H + Totals): rho = 0.3
      (correlated: if team wins big, over is more likely)
    - Same sport, different event: rho = 0.05
      (weakly correlated: league-wide trends)
    - Different sport: rho = 0.0 (independent)
    - Same event, same market: rho = 1.0 (duplicate — shouldn't happen)

    Parameters
    ----------
    base_variance : float
        Variance of a single binary bet outcome (~p*(1-p)).
        Default 0.25 corresponds to a 50/50 bet.
    """
    cov = np.eye(n_bets) * base_variance

    for i in range(n_bets):
        for j in range(i + 1, n_bets):
            if event_ids[i] == event_ids[j]:
                if markets[i] == markets[j]:
                    rho = 0.95  # near-duplicate
                else:
                    rho = 0.30  # same game, different market
            elif sports[i] == sports[j]:
                rho = 0.05  # same sport, weak league correlation
            else:
                rho = 0.0   # independent sports

            cov[i, j] = rho * base_variance
            cov[j, i] = rho * base_variance

    return cov
