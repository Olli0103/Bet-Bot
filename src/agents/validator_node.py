"""Cache-First Validator Node: quota-preserving odds re-verification.

The ValidatorNode decouples the Control Plane (agents) from the Data Plane
(API fetchers) by reading cached odds before spending API quota.

Workflow:
    1. Background ``fetch_scheduler`` continuously populates Redis with
       fresh odds for active sport keys.
    2. When validation is needed (tip_flow or human confirmation), this
       node reads Redis first.
    3. Only if the cache is stale (older than ``MAX_STALENESS_SECONDS``)
       does it spend exactly **one** targeted API call for the specific
       event — never the whole league.

This preserves API quota while ensuring no tip is confirmed at stale odds.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

# Maximum age (seconds) of cached odds before a live API call is permitted.
MAX_STALENESS_SECONDS = 30


async def get_validated_odds(
    event_id: str,
    selection: str,
    *,
    max_staleness: int = MAX_STALENESS_SECONDS,
) -> Tuple[Optional[float], bool]:
    """Return (current_odds, used_api_call) for a selection.

    Checks the Redis event-level cache first (populated by
    ``OddsFetcher.get_event_odds`` with 10 s TTL and by
    ``fetch_scheduler`` via sport-level caches).  Falls back to a
    targeted single-event API call only when data is stale.

    Parameters
    ----------
    event_id:
        The Odds-API event ID.
    selection:
        Outcome name (e.g. "Bayern Munich", "Over 2.5").
    max_staleness:
        Seconds before cached odds are considered stale.

    Returns
    -------
    (odds, used_api)
        ``odds`` is the decimal price for *selection*, or ``None`` if
        unavailable.  ``used_api`` is ``True`` when a fresh API call
        was made.
    """
    # --- 1. Try the short-lived per-event cache (set by get_event_odds) ---
    cache_key = f"odds:event:{event_id}"
    cached = cache.get_json(cache_key)
    if cached and isinstance(cached, dict):
        odds = cached.get(selection)
        if odds is not None and float(odds) > 1.0:
            log.debug(
                "validator: cache HIT for %s/%s (0 API calls)", event_id, selection,
            )
            return float(odds), False

    # --- 2. Try the timestamped validator cache (set by us on API hit) ---
    ts_key = f"odds:validated:{event_id}"
    ts_cached = cache.get_json(ts_key)
    if ts_cached and isinstance(ts_cached, dict):
        last_updated = ts_cached.get("_ts", 0)
        if (time.time() - last_updated) <= max_staleness:
            odds = ts_cached.get(selection)
            if odds is not None and float(odds) > 1.0:
                log.debug(
                    "validator: timestamped cache HIT for %s/%s (age %.0fs)",
                    event_id, selection, time.time() - last_updated,
                )
                return float(odds), False

    # --- 3. Cache miss or stale — spend exactly 1 targeted API call ---
    log.info(
        "validator: cache STALE for %s/%s — spending 1 API call",
        event_id, selection,
    )
    try:
        from src.integrations.odds_fetcher import OddsFetcher

        fetcher = OddsFetcher()
        fresh: Optional[Dict[str, float]] = await fetcher.get_event_odds(event_id)
        if fresh:
            # Persist with timestamp for subsequent checks within the window
            fresh_with_ts = dict(fresh)
            fresh_with_ts["_ts"] = time.time()
            cache.set_json(ts_key, fresh_with_ts, ttl_seconds=60)

            odds = fresh.get(selection)
            if odds is not None and float(odds) > 1.0:
                return float(odds), True
    except Exception as exc:
        log.warning("validator: API call failed for %s: %s", event_id, exc)

    return None, False


async def validate_before_placement(
    tip_id: str,
    event_id: str,
    selection: str,
    target_odds: float,
    model_probability: float,
    *,
    max_staleness: int = MAX_STALENESS_SECONDS,
) -> Dict:
    """Full slippage check invoked when the operator clicks [Platziert].

    Returns a dict with:
        ``valid``     – bool, whether the tip is still safe to place.
        ``live_odds`` – the odds used for validation (cached or fresh).
        ``used_api``  – whether an API call was spent.
        ``reason``    – human-readable explanation if rejected.
    """
    from src.core.betting_math import (
        DEFAULT_GERMAN_TAX_RATE,
        calculate_mao,
        tax_adjusted_expected_value,
    )

    live_odds, used_api = await get_validated_odds(
        event_id, selection, max_staleness=max_staleness,
    )

    if live_odds is None:
        # Cannot verify — let the operator's own eyes on Tipico be the
        # final validator (Bookie-Bypass pattern).  Record the original
        # target odds as the confirmed price.
        return {
            "valid": True,
            "live_odds": target_odds,
            "used_api": used_api,
            "reason": "Cache unavailable — operator confirms via bookmaker app",
        }

    # --- Math veto ---
    ev = tax_adjusted_expected_value(model_probability, live_odds, DEFAULT_GERMAN_TAX_RATE)
    mao = calculate_mao(model_probability, tax_rate=DEFAULT_GERMAN_TAX_RATE, required_edge=0.01)

    if ev <= 0:
        return {
            "valid": False,
            "live_odds": live_odds,
            "used_api": used_api,
            "reason": (
                f"Negative EV: {ev:+.4f} at live odds {live_odds:.3f} "
                f"(target was {target_odds:.3f})"
            ),
        }

    if live_odds < mao:
        return {
            "valid": False,
            "live_odds": live_odds,
            "used_api": used_api,
            "reason": (
                f"Odds {live_odds:.3f} below MAO {mao:.3f} "
                f"(target was {target_odds:.3f})"
            ),
        }

    return {
        "valid": True,
        "live_odds": live_odds,
        "used_api": used_api,
        "reason": "OK",
    }
