"""Spinning Wheel Guard: Tipico bet-acceptance delay handler.

Tipico's betslip takes 5-8 seconds to confirm ("spinning wheel").
Three possible responses:
  1. ``accepted``    — bet placed at original odds.
  2. ``odds_changed`` — re-offer at new (usually worse) odds.
  3. ``rejected``    — bet declined entirely (liquidity / limits).

This module caches the latest bet-placement response per event:selection
and checks whether re-offered odds still exceed the Minimum Acceptable
Odds (MAO).  If the re-offer is below MAO, the edge is gone and the
bet must be aborted to avoid negative-EV execution.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

# Cache keys
SPINNING_WHEEL_PREFIX = "tipico:bet_response:"
SPINNING_WHEEL_TTL = 3600  # 1 hour


def record_bet_response(
    event_id: str,
    selection: str,
    status: str,
    original_odds: float = 0.0,
    new_odds: float = 0.0,
) -> None:
    """Cache Tipico's bet-placement response for a given event:selection.

    Parameters
    ----------
    status : str
        One of ``"accepted"``, ``"odds_changed"``, ``"rejected"``.
    new_odds : float
        The re-offered odds (only meaningful when status == "odds_changed").
    """
    key = f"{SPINNING_WHEEL_PREFIX}{event_id}:{selection}"
    cache.set_json(key, {
        "status": status,
        "original_odds": original_odds,
        "new_odds": new_odds,
        "ts": time.time(),
    }, ttl_seconds=SPINNING_WHEEL_TTL)
    log.info(
        "Spinning wheel response recorded: %s %s → %s (orig=%.3f, new=%.3f)",
        event_id, selection, status, original_odds, new_odds,
    )


def check_spinning_wheel(
    event_id: str,
    selection: str,
    mao: float,
) -> Tuple[bool, str]:
    """Check if a recent Tipico response for this bet allows execution.

    Returns (ok, reason).  ``ok=True`` means proceed, ``ok=False`` means abort.

    Logic:
    - No cached response → assume first attempt, proceed.
    - ``accepted`` → proceed (already placed, skip re-execution via SETNX lock).
    - ``rejected`` → abort, Tipico declined.
    - ``odds_changed`` → check ``new_odds >= mao``.  If yes, proceed at
      re-offered price.  If no, abort — edge is gone.
    """
    key = f"{SPINNING_WHEEL_PREFIX}{event_id}:{selection}"
    data = cache.get_json(key)

    if not data:
        return True, ""  # No prior response — first attempt

    status = data.get("status", "")
    new_odds = float(data.get("new_odds", 0))

    if status == "accepted":
        return True, ""

    if status == "rejected":
        return False, f"Spinning wheel: Tipico rejected bet for {event_id}:{selection}"

    if status == "odds_changed":
        if new_odds <= 1.0:
            return False, f"Spinning wheel: invalid re-offer odds {new_odds}"
        if new_odds < mao:
            return False, (
                f"Spinning wheel: re-offered odds {new_odds:.3f} < MAO {mao:.3f} "
                f"— edge gone, bet aborted"
            )
        log.info(
            "Spinning wheel: re-offer accepted: %s %s new_odds=%.3f >= MAO=%.3f",
            event_id, selection, new_odds, mao,
        )
        return True, ""

    return True, ""  # Unknown status — proceed cautiously
