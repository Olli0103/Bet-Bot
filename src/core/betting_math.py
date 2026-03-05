from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.optimize import root_scalar


DEFAULT_GERMAN_TAX_RATE = 0.05
"""German Wettsteuer: 5% levied on gross payout (stake * odds) by regulated
bookmakers like Tipico.  This is the standard rate under the German Interstate
Treaty on Gambling (GlüStV 2021)."""


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def effective_tax_rate(
    base_tax: float = 0.05,
    tax_free_mode: bool = False,
    is_combo: bool = False,
    combo_legs: int = 0,
) -> float:
    """Compute effective tax rate accounting for Tipico tax-free promotions.

    Tipico occasionally offers tax-free bets for:
    - Specific combo promotions (typically 3+ legs)
    - Mobile-only promos
    - Event-specific campaigns

    When ``tax_free_mode`` is True the entire bet is untaxed.
    When ``is_combo`` and ``combo_legs >= 3`` a reduced tax rate applies
    (Tipico often halves or eliminates the tax on qualifying combos).
    """
    if tax_free_mode:
        return 0.0
    if is_combo and combo_legs >= 3:
        # Tipico's standard combo promotion: tax-free for 3+ leg combos
        return 0.0
    return base_tax


def get_net_payout(decimal_odds: float, tax_rate: float = DEFAULT_GERMAN_TAX_RATE) -> float:
    """Net payout per 1 unit stake after German gross-payout tax.

    The Wettsteuer is levied on (stake * odds), not on profit.
    Example: odds 2.0, tax 5% → payout = 2.0 * 0.95 = 1.90.
    """
    return decimal_odds * (1.0 - tax_rate)


def expected_value(
    model_probability: float,
    decimal_odds: float,
    tax_rate: float = 0.0,
) -> float:
    """EV per 1 unit stake.

    German betting tax (Tipico) is levied on the **gross payout**
    (stake * odds), not just the profit.  Example at odds 2.0 with
    5 % tax: payout = 2.0 * 0.95 = 1.90, net profit = 0.90 (not 0.95).
    """
    net_odds = decimal_odds * (1.0 - tax_rate)
    net_profit = net_odds - 1.0
    return model_probability * net_profit - (1.0 - model_probability)


def tax_adjusted_expected_value(
    model_probability: float,
    decimal_odds: float,
    tax_rate: float = DEFAULT_GERMAN_TAX_RATE,
) -> float:
    """EV calculation with German gross tax applied by default.

    Unlike ``expected_value`` which defaults to tax_rate=0.0 (requiring
    callers to remember the tax), this function defaults to the standard
    5% Wettsteuer.  Use this for Tippprovider-facing calculations where
    the tip accuracy must reflect what the end-user actually receives.
    """
    net_payout = get_net_payout(decimal_odds, tax_rate)
    return model_probability * (net_payout - 1.0) - (1.0 - model_probability)


def tax_adjusted_kelly_growth(
    model_probability: float,
    decimal_odds: float,
    fraction: float,
    tax_rate: float = DEFAULT_GERMAN_TAX_RATE,
) -> float:
    """Expected log-growth rate with German tax integrated into the utility.

    Standard Kelly maximises E[log(1 + f * net_profit)] which assumes
    zero transaction costs.  For German-regulated markets the 5% gross
    tax changes the payoff structure:

        G(f) = p * log(1 + f * (odds*(1-t) - 1)) + (1-p) * log(1 - f)

    This function returns G(f) so the caller can optimise *f* with the
    tax baked in, rather than applying tax as a post-hoc haircut.

    Parameters
    ----------
    model_probability : float
        Estimated true win probability.
    decimal_odds : float
        Bookmaker decimal odds (pre-tax).
    fraction : float
        Kelly fraction of bankroll to wager.
    tax_rate : float
        Gross-payout tax rate (default 5% German Wettsteuer).

    Returns
    -------
    float
        Expected log-growth rate.  Negative means the bet shrinks the
        bankroll in expectation.
    """
    if fraction <= 0.0 or fraction >= 1.0:
        return 0.0
    net_profit = get_net_payout(decimal_odds, tax_rate) - 1.0
    if net_profit <= 0.0:
        return -math.inf
    p = model_probability
    win_term = p * math.log(1.0 + fraction * net_profit)
    lose_term = (1.0 - p) * math.log(1.0 - fraction)
    return win_term + lose_term


def kelly_fraction(
    model_probability: float,
    decimal_odds: float,
    frac: float = 0.2,
    tax_rate: float = 0.0,
    max_fraction: float = 0.05,
) -> float:
    """Kelly criterion using net odds after tax.

    Returns fractional Kelly (scaled by ``frac``) with a **hard safety cap**
    at ``max_fraction`` (default 5% of bankroll).  This is a defense-in-depth
    measure: even if downstream caps (``apply_stake_cap``) have bugs or are
    bypassed, the math layer will never return a ruinous fraction.

    Use ``kelly_fraction_uncapped()`` for ranking/display where the true
    magnitude matters.
    """
    net_b = decimal_odds * (1.0 - tax_rate) - 1.0
    q = 1.0 - model_probability
    if net_b <= 0:
        return 0.0
    raw = (net_b * model_probability - q) / net_b
    scaled = max(0.0, raw) * frac
    return min(scaled, max_fraction)


def kelly_fraction_uncapped(
    model_probability: float,
    decimal_odds: float,
    frac: float = 0.2,
    tax_rate: float = 0.0,
) -> float:
    """Uncapped Kelly fraction for ranking/display (NOT for staking)."""
    net_b = decimal_odds * (1.0 - tax_rate) - 1.0
    q = 1.0 - model_probability
    if net_b <= 0:
        return 0.0
    raw = (net_b * model_probability - q) / net_b
    return max(0.0, raw) * frac


def kelly_stake(bankroll: float, kelly_f: float) -> float:
    return max(0.0, bankroll * kelly_f)


def calculate_mao(
    probability: float,
    tax_rate: float = 0.053,
    required_edge: float = 0.01,
) -> float:
    """Calculate the Minimum Acceptable Odds (MAO) for a bet.

    This is the break-even odds threshold accounting for tax and a
    minimum required edge.  If the bookmaker's live odds at execution
    time fall below this value, the bet must be aborted — the slippage
    has eaten the edge.

    Formula derivation:
        EV = prob * odds * (1 - tax) - 1.0 >= required_edge
        odds >= (1.0 + required_edge) / (prob * (1.0 - tax))

    Returns the minimum decimal odds needed to maintain positive EV.
    """
    if probability <= 0 or (1.0 - tax_rate) <= 0:
        return 999.0

    mao = (1.0 + required_edge) / (probability * (1.0 - tax_rate))
    return round(mao, 3)


def combo_odds(odds: Iterable[float]) -> float:
    out = 1.0
    for o in odds:
        out *= o
    return out


def combo_probability(probs: Iterable[float]) -> float:
    out = 1.0
    for p in probs:
        out *= p
    return out


def remove_vig_shin(prices: Dict[str, float]) -> Dict[str, float]:
    """Remove vig using Shin's Method — SOTA for skewed markets.

    Shin's method models the bookmaker's margin as arising from a fraction
    *z* of insider/sharp money in the market.  It solves for *z* such that
    the implied probabilities, adjusted for insider trading, sum to 1.0.

    This is strictly superior to the Power Method in markets with heavy
    favourite-longshot bias (e.g. 1.10 vs 8.00), because it correctly
    attributes more margin to the longshot side where insider information
    has less impact.

    Falls back to the Power Method if the Brentq solver fails to converge.
    """
    if not prices:
        return {}

    raw_ips = {sel: 1.0 / odds for sel, odds in prices.items() if odds > 1.0}
    if not raw_ips:
        return {}

    sum_pi = sum(raw_ips.values())
    if abs(sum_pi - 1.0) < 1e-9:
        return {sel: round(ip, 6) for sel, ip in raw_ips.items()}

    ips = list(raw_ips.values())
    sels = list(raw_ips.keys())

    def objective(z: float) -> float:
        total = 0.0
        for pi in ips:
            term = np.sqrt(z ** 2 + 4 * (1 - z) * (pi ** 2) / sum_pi)
            p_i = (term - z) / (2 * (1 - z))
            total += p_i
        return total - 1.0

    try:
        sol = root_scalar(objective, bracket=[1e-6, 1.0 - 1e-6], method="brentq")
        z = sol.root
    except (ValueError, RuntimeError):
        # Shin didn't converge — fall back to Power Method
        return _remove_vig_power(prices)

    fair = {}
    for i, sel in enumerate(sels):
        pi = ips[i]
        term = np.sqrt(z ** 2 + 4 * (1 - z) * (pi ** 2) / sum_pi)
        p_i = (term - z) / (2 * (1 - z))
        fair[sel] = round(p_i, 6)

    return fair


def _remove_vig_power(prices: Dict[str, float]) -> Dict[str, float]:
    """Remove vig using the Power Method (fallback for Shin).

    Solves ``sum(ip_i ^ k) = 1`` for exponent ``k`` via Newton's method.
    """
    if not prices:
        return {}
    raw_ips = {}
    for sel, odds in prices.items():
        if odds > 1.0:
            raw_ips[sel] = 1.0 / odds
    if not raw_ips:
        return {}
    total_ip = sum(raw_ips.values())
    if total_ip <= 0:
        return {}

    if abs(total_ip - 1.0) < 1e-9:
        return {sel: round(ip, 6) for sel, ip in raw_ips.items()}

    k = 1.0
    ips = list(raw_ips.values())
    for _ in range(50):
        s = sum(p ** k for p in ips)
        if abs(s - 1.0) < 1e-12:
            break
        ds = sum(p ** k * math.log(p) for p in ips if p > 0)
        if abs(ds) < 1e-15:
            break
        k -= (s - 1.0) / ds
        k = max(0.1, min(k, 10.0))

    fair = {}
    sels = list(raw_ips.keys())
    for i, sel in enumerate(sels):
        fair[sel] = round(ips[i] ** k, 6)

    total_fair = sum(fair.values())
    if total_fair > 0 and abs(total_fair - 1.0) > 1e-6:
        fair = {sel: round(p / total_fair, 6) for sel, p in fair.items()}

    return fair


def _remove_vig(prices: Dict[str, float]) -> Dict[str, float]:
    """Remove vig — dispatches to Shin's Method (primary) or Power (fallback)."""
    return remove_vig_shin(prices)


def public_bias_score(
    sharp_prices: Dict[str, float],
    retail_prices: Dict[str, float],
) -> Dict[str, float]:
    """Detect Tipico market shading (public bias) vs sharp book.

    Compares **vig-removed fair probabilities** between Pinnacle (sharp) and
    Tipico (retail).  Both books' margins are stripped before comparison so
    the bias score reflects genuine shading, not margin differences.

    Returns a per-selection bias score:
    - bias > 0.02: Tipico is shading this selection (public over-bet)
    - bias < -0.02: Tipico is offering relative value on this selection
    """
    if not sharp_prices or not retail_prices:
        return {}

    # Remove vig from BOTH books before comparing
    sharp_fair = _remove_vig(sharp_prices)
    retail_fair = _remove_vig(retail_prices)

    bias: Dict[str, float] = {}
    for sel in sharp_fair:
        if sel not in retail_fair:
            continue
        # Positive gap = retail implies higher fair prob than sharp → over-bet
        bias[sel] = round(retail_fair[sel] - sharp_fair[sel], 4)

    return bias
