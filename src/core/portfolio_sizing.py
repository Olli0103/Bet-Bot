"""Simultaneous Kelly Portfolio Sizing (Mean-Variance + CVaR Optimization).

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

**CVaR (Conditional Value at Risk):**
Betting returns have fat left tails — a portfolio can lose most of its
simultaneous stakes in a single bad window.  Pure mean-variance ignores
this asymmetry.  The ``cvar_constrained_kelly_sizing`` function adds a
CVaR constraint: the expected loss in the worst α% of scenarios must
not exceed ``max_cvar_loss`` (fraction of bankroll).  This caps left-
tail risk without sacrificing much upside.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)

# Maximum fraction of bankroll that can be risked across all simultaneous bets
MAX_TOTAL_EXPOSURE = 0.15  # 15% hard cap

# CVaR defaults
CVAR_ALPHA = 0.05           # 5th-percentile tail
CVAR_MAX_LOSS = 0.08        # max 8% expected loss in worst 5% scenarios
CVAR_N_SCENARIOS = 5_000    # Monte Carlo scenarios


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


# ---------------------------------------------------------------------------
# CVaR-constrained portfolio sizing
# ---------------------------------------------------------------------------

def _simulate_portfolio_returns(
    fractions: np.ndarray,
    probabilities: np.ndarray,
    odds: np.ndarray,
    cov_matrix: np.ndarray,
    n_scenarios: int = CVAR_N_SCENARIOS,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Monte Carlo simulation of portfolio returns.

    Each scenario draws correlated Bernoulli outcomes (win/loss) for
    every bet, then computes the portfolio return as:
        R = sum_i f_i * (odds_i - 1)  if bet i wins
          - sum_i f_i                  if bet i loses

    Correlation is introduced via a Gaussian copula: draw correlated
    normals from the covariance matrix, then threshold each at the
    bet's win probability to decide win/loss.

    Returns
    -------
    np.ndarray
        Portfolio returns for each scenario, shape (n_scenarios,).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_bets = len(fractions)

    # Correlation matrix from covariance
    std = np.sqrt(np.diag(cov_matrix))
    std[std < 1e-12] = 1e-12
    corr = cov_matrix / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)

    # Ensure PSD for Cholesky
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 1e-8:
        corr = corr + np.eye(n_bets) * (1e-8 - eigvals.min())

    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Fallback: independent draws
        L = np.eye(n_bets)

    # Draw correlated standard normals
    z = rng.standard_normal((n_scenarios, n_bets))
    correlated = z @ L.T  # (n_scenarios, n_bets)

    # Convert to correlated uniform via CDF, then to Bernoulli
    from scipy.stats import norm
    u = norm.cdf(correlated)
    wins = u < probabilities  # (n_scenarios, n_bets) boolean

    # Compute returns per scenario
    payoffs = np.where(wins, fractions * (odds - 1.0), -fractions)
    portfolio_returns = payoffs.sum(axis=1)

    return portfolio_returns


def compute_cvar(
    returns: np.ndarray,
    alpha: float = CVAR_ALPHA,
) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

    CVaR_α = mean of the worst α fraction of returns.
    A more negative CVaR means higher left-tail risk.

    Parameters
    ----------
    returns : np.ndarray
        Simulated portfolio returns.
    alpha : float
        Tail probability (default 0.05 = worst 5%).

    Returns
    -------
    float
        CVaR as a negative number (loss).  E.g. -0.08 means the
        expected loss in the worst 5% of scenarios is 8% of bankroll.
    """
    n = len(returns)
    if n == 0:
        return 0.0
    cutoff = max(1, int(np.ceil(n * alpha)))
    sorted_returns = np.sort(returns)
    return float(sorted_returns[:cutoff].mean())


def cvar_constrained_kelly_sizing(
    edges: np.ndarray,
    cov_matrix: np.ndarray,
    probabilities: np.ndarray,
    odds: np.ndarray,
    kelly_fraction: float = 0.25,
    max_total_exposure: float = MAX_TOTAL_EXPOSURE,
    alpha: float = CVAR_ALPHA,
    max_cvar_loss: float = CVAR_MAX_LOSS,
    n_scenarios: int = CVAR_N_SCENARIOS,
) -> np.ndarray:
    """Kelly sizing with a CVaR constraint on left-tail risk.

    First runs standard mean-variance Kelly.  Then checks whether
    the resulting portfolio satisfies the CVaR constraint via Monte
    Carlo simulation.  If not, iteratively scales down until the
    constraint is met.

    Parameters
    ----------
    edges : np.ndarray
        Expected value per bet, shape (n_bets,).
    cov_matrix : np.ndarray
        Covariance matrix, shape (n_bets, n_bets).
    probabilities : np.ndarray
        Win probabilities per bet, shape (n_bets,).
    odds : np.ndarray
        Decimal odds per bet, shape (n_bets,).
    kelly_fraction : float
        Fractional Kelly multiplier.
    max_total_exposure : float
        Hard cap on total allocation.
    alpha : float
        CVaR tail probability (default 0.05).
    max_cvar_loss : float
        Maximum acceptable CVaR loss as a positive fraction of bankroll.
        E.g. 0.08 = "the expected loss in the worst 5% of scenarios must
        not exceed 8% of bankroll".
    n_scenarios : int
        Monte Carlo scenarios for CVaR estimation.

    Returns
    -------
    np.ndarray
        Optimal fractions satisfying both exposure and CVaR constraints.
    """
    n_bets = len(edges)
    if n_bets == 0:
        return np.array([])

    probabilities = np.asarray(probabilities, dtype=float)
    odds = np.asarray(odds, dtype=float)

    # Step 1: Get unconstrained (mean-variance) Kelly solution
    fractions = simultaneous_kelly_sizing(
        edges, cov_matrix, kelly_fraction, max_total_exposure,
    )

    if fractions.sum() < 1e-8:
        return fractions  # Nothing to constrain

    # Step 2: Check CVaR constraint via Monte Carlo
    rng = np.random.default_rng(42)
    returns = _simulate_portfolio_returns(
        fractions, probabilities, odds, cov_matrix, n_scenarios, rng,
    )
    cvar = compute_cvar(returns, alpha)

    if cvar >= -max_cvar_loss:
        log.debug(
            "CVaR constraint satisfied: CVaR(%.0f%%)=%.4f >= -%.4f",
            alpha * 100, cvar, max_cvar_loss,
        )
        return fractions

    # Step 3: Iteratively scale down to satisfy CVaR constraint
    # Binary search for the largest scale factor that satisfies CVaR
    lo, hi = 0.0, 1.0
    best_fractions = fractions * 0.0  # worst case: zero everything

    for _ in range(20):  # 20 iterations gives ~1e-6 precision
        mid = (lo + hi) / 2.0
        scaled = fractions * mid
        returns = _simulate_portfolio_returns(
            scaled, probabilities, odds, cov_matrix, n_scenarios, rng,
        )
        trial_cvar = compute_cvar(returns, alpha)

        if trial_cvar >= -max_cvar_loss:
            lo = mid
            best_fractions = scaled
        else:
            hi = mid

    best_fractions[best_fractions < 1e-6] = 0.0
    scale_applied = lo

    log.info(
        "CVaR constraint required scaling: %.1f%% of original Kelly "
        "(CVaR(%.0f%%)=%.4f, limit=%.4f)",
        scale_applied * 100, alpha * 100,
        compute_cvar(
            _simulate_portfolio_returns(
                best_fractions, probabilities, odds, cov_matrix,
                n_scenarios, rng,
            ),
            alpha,
        ),
        -max_cvar_loss,
    )

    return np.round(best_fractions, 6)


def build_covariance_matrix(
    n_bets: int,
    event_ids: List[str],
    markets: List[str],
    sports: List[str],
    base_variance: float = 0.25,
    historical_residuals: Optional[np.ndarray] = None,
    shrinkage_target: str = "heuristic",
) -> np.ndarray:
    """Build a covariance matrix from bet metadata.

    When ``historical_residuals`` are provided, uses the **Ledoit-Wolf
    empirical shrinkage estimator** to compute correlations from actual
    bet outcome data, shrinking toward the heuristic prior to avoid
    overfitting on small samples.  This prevents the portfolio from
    over-exposing the user to a single league or outcome type.

    Falls back to pure heuristic correlations when no historical data
    is available (cold-start).

    Heuristic correlation rules (prior / fallback):
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
    historical_residuals : np.ndarray, optional
        Matrix of shape (n_observations, n_bets) containing historical
        bet outcome residuals (actual - predicted).  When provided,
        the empirical covariance is estimated and shrunk toward the
        heuristic prior using Ledoit-Wolf optimal shrinkage.
    shrinkage_target : str
        If "heuristic" (default), shrink toward the heuristic
        correlation matrix.  If "identity", shrink toward identity
        (standard Ledoit-Wolf).
    """
    # --- Heuristic prior covariance (always computed) ---
    heuristic_cov = np.eye(n_bets) * base_variance

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

            heuristic_cov[i, j] = rho * base_variance
            heuristic_cov[j, i] = rho * base_variance

    # --- Empirical shrinkage (Ledoit-Wolf) when historical data exists ---
    if historical_residuals is not None and historical_residuals.shape[0] >= n_bets * 2:
        try:
            empirical_cov = _ledoit_wolf_shrinkage(
                historical_residuals, heuristic_cov, shrinkage_target,
            )
            return empirical_cov
        except Exception:
            log.warning(
                "Ledoit-Wolf shrinkage failed, falling back to heuristic covariance"
            )

    return heuristic_cov


def _ledoit_wolf_shrinkage(
    residuals: np.ndarray,
    prior_cov: np.ndarray,
    target: str = "heuristic",
) -> np.ndarray:
    """Compute Ledoit-Wolf shrinkage estimate of the covariance matrix.

    Optimal shrinkage intensity α is computed analytically:
        Σ_shrunk = α * F + (1 - α) * S

    where S is the sample covariance, F is the shrinkage target (either
    the heuristic prior or the identity scaled to match trace), and α
    is chosen to minimise expected squared Frobenius loss.

    Parameters
    ----------
    residuals : np.ndarray
        (n_obs, n_features) matrix of centred residuals.
    prior_cov : np.ndarray
        Heuristic prior covariance matrix (shrinkage target when
        ``target="heuristic"``).
    target : str
        "heuristic" to shrink toward ``prior_cov``, "identity" for
        scaled identity.

    Returns
    -------
    np.ndarray
        Shrinkage-estimated covariance matrix.
    """
    n_obs, n_features = residuals.shape
    # Centre residuals
    residuals = residuals - residuals.mean(axis=0)
    # Sample covariance
    sample_cov = np.dot(residuals.T, residuals) / n_obs

    if target == "identity":
        mu = np.trace(sample_cov) / n_features
        shrinkage_target = mu * np.eye(n_features)
    else:
        shrinkage_target = prior_cov

    # Compute optimal shrinkage intensity (Oracle Approximating Shrinkage)
    delta = sample_cov - shrinkage_target
    # Squared Frobenius norm of delta
    delta_sq = np.sum(delta ** 2)

    if delta_sq < 1e-12:
        return sample_cov  # sample ≈ target already

    # Estimate variance of sample covariance entries
    # Using the Ledoit-Wolf (2004) formula for the optimal alpha
    sum_sq = 0.0
    for k in range(n_obs):
        x_k = residuals[k:k+1, :]  # (1, n_features)
        outer_k = np.dot(x_k.T, x_k)
        diff_k = outer_k - sample_cov
        sum_sq += np.sum(diff_k ** 2)
    beta = sum_sq / (n_obs ** 2)

    # Shrinkage intensity: clamp to [0, 1]
    alpha = min(1.0, max(0.0, beta / delta_sq))

    log.info(
        "Ledoit-Wolf shrinkage: alpha=%.4f (%.0f%% toward %s prior)",
        alpha, alpha * 100, target,
    )

    return alpha * shrinkage_target + (1.0 - alpha) * sample_cov
