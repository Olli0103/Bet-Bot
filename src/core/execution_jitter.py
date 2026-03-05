"""Humanized Execution Jitter: prevents WAF bot-detection bans.

When 5+ signals fire simultaneously (e.g. Bundesliga Saturday 15:30),
the worker loop would normally blast 5 POST requests within 200ms.
No human clicks that fast — Akamai/Cloudflare WAFs instantly flag this
pattern and return HTTP 429 or shadow-ban the account.

This module adds randomized, human-like delays between consecutive
bet placements so the execution pattern mimics real user behavior:

  - Base delay: 1.5 - 3.5 seconds between bets (random uniform)
  - Burst detection: if > 2 bets in 10 seconds, insert a 5-8s cooldown
  - Jitter: additional ±0.3s Gaussian noise on every delay

The delays are only applied to REAL execution requests, not to
analysis/evaluation.  Paper bets skip the jitter entirely.
"""
from __future__ import annotations

import logging
import random
import time
from collections import deque
from typing import Tuple

log = logging.getLogger(__name__)

# Configurable parameters
MIN_DELAY_SECONDS = 1.5
MAX_DELAY_SECONDS = 3.5
BURST_COOLDOWN_MIN = 5.0
BURST_COOLDOWN_MAX = 8.0
BURST_THRESHOLD = 2          # bets in window
BURST_WINDOW_SECONDS = 10.0  # sliding window
JITTER_SIGMA = 0.3           # Gaussian noise std dev

# Track recent execution timestamps for burst detection
_recent_executions: deque = deque(maxlen=50)


def _prune_old_entries(window: float) -> None:
    """Remove entries older than the burst window."""
    cutoff = time.monotonic() - window
    while _recent_executions and _recent_executions[0] < cutoff:
        _recent_executions.popleft()


def get_execution_delay() -> Tuple[float, str]:
    """Calculate the next humanized delay before bet execution.

    Returns (delay_seconds, reason).
    """
    _prune_old_entries(BURST_WINDOW_SECONDS)
    recent_count = len(_recent_executions)

    if recent_count >= BURST_THRESHOLD:
        # Burst detected — insert longer cooldown
        base = random.uniform(BURST_COOLDOWN_MIN, BURST_COOLDOWN_MAX)
        jitter = random.gauss(0, JITTER_SIGMA)
        delay = max(0.5, base + jitter)
        return delay, f"burst_cooldown (n={recent_count} in {BURST_WINDOW_SECONDS}s)"

    # Normal human-like delay
    base = random.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS)
    jitter = random.gauss(0, JITTER_SIGMA)
    delay = max(0.3, base + jitter)
    return delay, "normal_jitter"


def record_execution() -> None:
    """Record that a bet was just executed (for burst detection)."""
    _recent_executions.append(time.monotonic())


async def apply_execution_jitter() -> float:
    """Wait for a humanized delay before executing.

    Returns the actual delay applied (seconds).
    """
    import asyncio

    delay, reason = get_execution_delay()
    log.debug("Execution jitter: %.2fs (%s)", delay, reason)
    await asyncio.sleep(delay)
    record_execution()
    return delay
