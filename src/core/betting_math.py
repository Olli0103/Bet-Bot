from __future__ import annotations

from typing import Iterable


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def expected_value(model_probability: float, decimal_odds: float, tax_rate: float = 0.0) -> float:
    """EV per 1 unit stake. tax_rate adjusts gross profit (e.g., 0.05 for Tipico 5% tax)."""
    gross_profit = decimal_odds - 1.0
    net_profit = gross_profit * (1.0 - tax_rate)
    return model_probability * net_profit - (1.0 - model_probability)


def kelly_fraction(model_probability: float, decimal_odds: float, frac: float = 0.2, tax_rate: float = 0.0) -> float:
    """Kelly criterion using net odds after tax."""
    gross_b = decimal_odds - 1.0
    net_b = gross_b * (1.0 - tax_rate)
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
