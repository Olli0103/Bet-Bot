from __future__ import annotations

from typing import Dict, Iterable, Optional


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


def _remove_vig(prices: Dict[str, float]) -> Dict[str, float]:
    """Remove vig from a set of odds using the proportional-to-implied method.

    Converts raw implied probabilities to fair probabilities that sum to 1.0.
    This is essential before comparing two books with different margins.
    """
    if not prices:
        return {}
    raw_ips = {}
    for sel, odds in prices.items():
        if odds > 1.0:
            raw_ips[sel] = 1.0 / odds
    total_ip = sum(raw_ips.values())
    if total_ip <= 0:
        return {}
    return {sel: round(ip / total_ip, 6) for sel, ip in raw_ips.items()}


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
