from __future__ import annotations

from typing import Iterable


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def expected_value(model_probability: float, decimal_odds: float) -> float:
    # EV per 1 unit stake
    return model_probability * (decimal_odds - 1.0) - (1.0 - model_probability)


def kelly_fraction(model_probability: float, decimal_odds: float, frac: float = 0.2) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - model_probability
    if b <= 0:
        return 0.0
    raw = (b * model_probability - q) / b
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
