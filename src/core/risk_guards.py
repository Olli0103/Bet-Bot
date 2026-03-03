"""Risk Guards: Confidence gates + stake caps.

Central module so both live_feed signal generation and the executioner agent
apply identical rules.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from src.core.settings import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence gate
# ---------------------------------------------------------------------------

def get_min_confidence(sport: str, market: str = "h2h") -> float:
    """Return the minimum model_probability required to allow a bet.

    Resolves by sport prefix + market type, falling back to the default.
    """
    s = sport.lower()
    m = market.lower()

    if s.startswith(("soccer", "football")):
        if "totals" in m:
            return settings.min_confidence_soccer_totals
        if "spread" in m:
            return settings.min_confidence_soccer_spread
        return settings.min_confidence_soccer_h2h
    if s.startswith("tennis"):
        return settings.min_confidence_tennis
    if s.startswith("basketball"):
        return settings.min_confidence_basketball
    if s.startswith("icehockey"):
        return settings.min_confidence_icehockey
    if s.startswith("americanfootball"):
        return settings.min_confidence_americanfootball
    return settings.min_confidence_default


def passes_confidence_gate(
    model_probability: float,
    sport: str,
    market: str = "h2h",
) -> Tuple[bool, float]:
    """Check if model_probability meets the confidence gate.

    Returns (passed, min_conf) so callers can log the reason.
    """
    min_conf = get_min_confidence(sport, market)
    return model_probability >= min_conf, min_conf


# ---------------------------------------------------------------------------
# Stake caps
# ---------------------------------------------------------------------------

def apply_stake_cap(
    stake: float,
    bankroll: float,
    odds: float = 1.0,
    market: str = "h2h",
    selection: str = "",
) -> Tuple[float, bool]:
    """Enforce hard stake caps.

    Returns (capped_stake, was_capped).

    Rules:
    - General cap: max_stake_pct * bankroll (default 1.5%)
    - Draw/longshot cap: max_stake_longshot_pct * bankroll (default 0.75%)
      applies when odds >= longshot_odds_threshold OR selection contains 'Draw'
    """
    is_draw = "draw" in selection.lower() or "unentschieden" in selection.lower()
    is_longshot = odds >= settings.longshot_odds_threshold

    if is_draw or is_longshot:
        cap = bankroll * settings.max_stake_longshot_pct
    else:
        cap = bankroll * settings.max_stake_pct

    if stake > cap:
        return round(cap, 2), True
    return round(stake, 2), False
