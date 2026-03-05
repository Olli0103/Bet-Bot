"""Proactive Session Manager: bookie token refresh before sniper window.

Tipico (and other bookies) use session tokens that expire after ~30 minutes
of inactivity.  If the token expires during the sniper window (< 15 min
before kickoff), the bot gets a 401 and misses the trade.

This module tracks session freshness and proactively refreshes tokens
BEFORE the sniper window opens, ensuring the session is hot when we need
to fire.

Architecture:
  - ``ensure_session_fresh()`` is called by the orchestrator before each
    sniper-mode cycle.
  - If the session was last refreshed > ``MAX_SESSION_AGE_SECONDS`` ago,
    it triggers a lightweight keepalive request (e.g. GET /account/balance)
    that resets the server-side session timer.
  - A SETNX-based lock prevents multiple workers from refreshing
    simultaneously.
"""
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

# Session state keys
SESSION_STATE_KEY = "bookie:session:state"
SESSION_REFRESH_LOCK = "bookie:session:refresh_lock"

# Refresh if session is older than 20 minutes (Tipico timeout ~30 min)
MAX_SESSION_AGE_SECONDS = 20 * 60  # 20 minutes
# Lock TTL for refresh operation
REFRESH_LOCK_TTL = 30  # 30 seconds


def get_session_age() -> Optional[float]:
    """Return seconds since last session refresh, or None if unknown."""
    state = cache.get_json(SESSION_STATE_KEY)
    if not state:
        return None
    last_refresh = state.get("last_refresh_ts", 0)
    if last_refresh <= 0:
        return None
    return time.time() - last_refresh


def record_session_refresh(source: str = "proactive") -> None:
    """Record that the bookie session was just refreshed."""
    cache.set_json(SESSION_STATE_KEY, {
        "last_refresh_ts": time.time(),
        "source": source,
    }, ttl_seconds=3600)


def needs_refresh() -> bool:
    """Check if the session needs a proactive refresh."""
    age = get_session_age()
    if age is None:
        return True  # Unknown state — refresh to be safe
    return age > MAX_SESSION_AGE_SECONDS


def ensure_session_fresh(
    refresh_fn=None,
) -> Tuple[bool, str]:
    """Proactively refresh the bookie session if it's getting stale.

    Parameters
    ----------
    refresh_fn : callable, optional
        A function that performs the actual session keepalive (e.g.
        ``GET /account/balance``).  If not provided, only the staleness
        check is performed and the result indicates whether a refresh
        was needed.

    Returns
    -------
    (refreshed, reason) : (bool, str)
        ``refreshed=True`` if a refresh was performed or not needed.
        ``refreshed=False`` if the refresh failed or was skipped (lock held).
    """
    if not needs_refresh():
        return True, "session_fresh"

    # Acquire refresh lock to prevent thundering herd
    if not cache.setnx(SESSION_REFRESH_LOCK, "refreshing", ttl_seconds=REFRESH_LOCK_TTL):
        log.debug("Session refresh lock held by another worker")
        return True, "refresh_in_progress"

    try:
        if refresh_fn is not None:
            refresh_fn()
        record_session_refresh(source="proactive")
        log.info("Bookie session proactively refreshed (was %.0fs old)", get_session_age() or 0)
        return True, "refreshed"
    except Exception as exc:
        log.warning("Proactive session refresh failed: %s", exc)
        return False, f"refresh_failed: {exc}"
    finally:
        # Release the lock
        cache.delete(SESSION_REFRESH_LOCK)
