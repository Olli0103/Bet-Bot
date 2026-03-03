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
    """EV per 1 unit stake. tax_rate adjusts gross profit (e.g., 0.05 for Tipico 5% tax)."""
    gross_profit = decimal_odds - 1.0
    net_profit = gross_profit * (1.0 - tax_rate)
    return model_probability * net_profit - (1.0 - model_probability)


def kelly_fraction(
    model_probability: float,
    decimal_odds: float,
    frac: float = 0.2,
    tax_rate: float = 0.0,
    max_fraction: float = 0.05,
) -> float:
    """Kelly criterion using net odds after tax.

    Hard-capped at ``max_fraction`` (default 5 %) of bankroll to prevent
    a single bet from risking too much, regardless of model confidence.
    """
    gross_b = decimal_odds - 1.0
    net_b = gross_b * (1.0 - tax_rate)
    q = 1.0 - model_probability
    if net_b <= 0:
        return 0.0
    raw = (net_b * model_probability - q) / net_b
    return min(max(0.0, raw) * frac, max_fraction)


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


def public_bias_score(
    sharp_prices: Dict[str, float],
    retail_prices: Dict[str, float],
) -> Dict[str, float]:
    """Detect Tipico market shading (public bias) vs sharp book.

    When Tipico shortens the favorite's odds more than the underdog relative
    to Pinnacle, the favorite carries extra retail-driven vig. Returns a
    per-selection bias score: positive = Tipico is shading this selection
    (over-bet by the public), negative = Tipico is offering relative value.

    Interpretation:
    - bias > 0.02: Tipico favorite is significantly shaded → higher skepticism
    - bias < -0.02: Tipico underdog is relatively generous → potential value
    """
    if not sharp_prices or not retail_prices:
        return {}

    # Compute implied probability gap per selection
    bias: Dict[str, float] = {}
    for sel in sharp_prices:
        sharp_odds = sharp_prices.get(sel, 0)
        retail_odds = retail_prices.get(sel, 0)
        if sharp_odds <= 1.0 or retail_odds <= 1.0:
            continue
        sharp_ip = 1.0 / sharp_odds
        retail_ip = 1.0 / retail_odds
        # Positive gap = retail implies higher probability than sharp → over-bet
        bias[sel] = round(retail_ip - sharp_ip, 4)

    return bias
