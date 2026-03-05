"""Sequential, rate-limit-aware odds fetch scheduler.

Replaces the naive parallel-burst fetch pattern with a controlled queue:
- One request per sport_key at a time
- Configurable inter-request delay with jitter (800-1500ms default)
- 429-aware retry with Retry-After header support
- Outright/futures key filtering
- Per-key cooldown on repeated failures (422/429)
- Staleness-aware cache fallback during rate limiting
- In-play terminator: live matches are excluded from sniper scheduling
"""
from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.core.settings import settings
from src.core.source_health import record_failure, record_success, is_available
from src.data.redis_cache import cache
from src.integrations.odds_fetcher import OddsFetcher, get_trading_window_end

log = logging.getLogger(__name__)

# Outright / futures patterns to exclude from the normal match pipeline
_OUTRIGHT_PATTERNS = re.compile(
    r"_(winner|championship_winner|futures|outright|mvp|"
    r"top_scorer|relegation|promotion)$",
    re.IGNORECASE,
)

# Default scheduling parameters
DEFAULT_MIN_DELAY_MS = 800
DEFAULT_MAX_DELAY_MS = 1500
DEFAULT_MAX_RETRIES = 3
COOLDOWN_KEY_PREFIX = "fetch:cooldown:"
FETCH_STATS_KEY = "fetch:cycle_stats"


def is_outright_key(sport_key: str) -> bool:
    """Check if a sport key represents an outright/futures market."""
    return bool(_OUTRIGHT_PATTERNS.search(sport_key))


def filter_match_keys(sport_keys: List[str]) -> Tuple[List[str], List[str]]:
    """Split sport keys into (match_keys, outright_keys)."""
    match_keys = []
    outright_keys = []
    for k in sport_keys:
        if is_outright_key(k):
            outright_keys.append(k)
        else:
            match_keys.append(k)
    if outright_keys:
        log.info("fetch_scheduler: filtered %d outright keys: %s",
                 len(outright_keys), outright_keys[:5])
    return match_keys, outright_keys


def _is_key_cooled_down(sport_key: str) -> bool:
    """Check if a sport key is in cooldown due to repeated failures."""
    cd = cache.get_json(f"{COOLDOWN_KEY_PREFIX}{sport_key}")
    if not cd:
        return False
    until = cd.get("until", 0)
    return time.time() < until


def _set_key_cooldown(sport_key: str, seconds: int = 300):
    """Put a sport key into cooldown after repeated failures."""
    cache.set_json(
        f"{COOLDOWN_KEY_PREFIX}{sport_key}",
        {"until": time.time() + seconds, "reason": "repeated_failure"},
        ttl_seconds=seconds + 10,
    )
    log.warning("fetch_scheduler: key %s cooled down for %ds", sport_key, seconds)


class FetchCycleStats:
    """Tracks per-cycle fetch statistics for the source gap report."""

    def __init__(self):
        self.fetched_keys: List[str] = []
        self.skipped_outright: List[str] = []
        self.skipped_cooldown: List[str] = []
        self.status_counts: Dict[str, int] = {}  # "200": N, "429": N, etc.
        self.raw_events_by_sport: Dict[str, int] = {}
        self.total_raw_events: int = 0
        self.errors: List[Dict[str, str]] = []
        self.started_at: float = time.time()
        self.finished_at: float = 0.0

    def record_status(self, status_code: int):
        key = str(status_code)
        self.status_counts[key] = self.status_counts.get(key, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        self.finished_at = time.time()
        return {
            "fetched_keys": self.fetched_keys,
            "skipped_outright": self.skipped_outright,
            "skipped_cooldown": self.skipped_cooldown,
            "status_counts": self.status_counts,
            "raw_events_by_sport": self.raw_events_by_sport,
            "total_raw_events": self.total_raw_events,
            "errors": self.errors[-20:],  # keep last 20
            "duration_sec": round(self.finished_at - self.started_at, 2),
        }


def sequential_fetch_odds(
    sport_keys: List[str],
    odds_fetcher: Optional[OddsFetcher] = None,
    regions: str = "eu",
    markets: str = "h2h,spreads,totals",
    ttl_seconds: int = 120,
    min_delay_ms: int = DEFAULT_MIN_DELAY_MS,
    max_delay_ms: int = DEFAULT_MAX_DELAY_MS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> Tuple[List[Tuple[str, Dict[str, Any]]], FetchCycleStats]:
    """Fetch odds sequentially with rate-limit awareness.

    Returns (raw_events, cycle_stats) where raw_events is a list of
    (sport_key, event_dict) tuples.
    """
    fetcher = odds_fetcher or OddsFetcher()
    stats = FetchCycleStats()

    # Filter outrights
    match_keys, outright_keys = filter_match_keys(sport_keys)
    stats.skipped_outright = outright_keys

    raw_events: List[Tuple[str, Dict[str, Any]]] = []

    if not is_available("odds_api"):
        log.warning("fetch_scheduler: odds_api circuit breaker open, using cache fallback")
        # Try to return cached data
        for sport in match_keys:
            cached = cache.get_json(f"odds:{sport}:{regions}:{markets}")
            if cached and isinstance(cached, list):
                for e in cached:
                    raw_events.append((sport, e))
                stats.raw_events_by_sport[sport] = len(cached)
                stats.total_raw_events += len(cached)
        return raw_events, stats

    for i, sport in enumerate(match_keys):
        # Check per-key cooldown
        if _is_key_cooled_down(sport):
            stats.skipped_cooldown.append(sport)
            # Still try cached data
            cached = cache.get_json(f"odds:{sport}:{regions}:{markets}")
            if cached and isinstance(cached, list):
                for e in cached:
                    raw_events.append((sport, e))
                stats.raw_events_by_sport[sport] = len(cached)
                stats.total_raw_events += len(cached)
            continue

        # Inter-request delay (skip for first request)
        if i > 0:
            delay = random.randint(min_delay_ms, max_delay_ms) / 1000.0
            time.sleep(delay)

        # Fetch with retry
        success = False
        for attempt in range(max_retries):
            try:
                events = fetcher.get_sport_odds(
                    sport_key=sport,
                    regions=regions,
                    markets=markets,
                    ttl_seconds=ttl_seconds,
                )
                stats.record_status(200)
                record_success("odds_api")

                if isinstance(events, list):
                    for e in events:
                        raw_events.append((sport, e))
                    stats.raw_events_by_sport[sport] = len(events)
                    stats.total_raw_events += len(events)
                    stats.fetched_keys.append(sport)

                success = True
                break

            except Exception as exc:
                exc_str = str(exc)

                # Detect HTTP status from exception message
                status_code = _extract_status_code(exc_str)
                stats.record_status(status_code or 0)

                if status_code == 429:
                    # Rate limited - respect Retry-After or exponential backoff
                    retry_after = _extract_retry_after(exc_str)
                    wait_time = retry_after if retry_after else (2 ** (attempt + 1)) + random.random()
                    log.warning(
                        "fetch_scheduler: 429 rate limit for %s, waiting %.1fs (attempt %d/%d)",
                        sport, wait_time, attempt + 1, max_retries,
                    )
                    time.sleep(wait_time)
                    record_failure("odds_api", f"429 rate limit: {sport}")
                    continue

                elif status_code == 422:
                    # Invalid request - cooldown this key
                    log.warning("fetch_scheduler: 422 for %s, putting in cooldown", sport)
                    _set_key_cooldown(sport, seconds=600)
                    record_failure("odds_api", f"422 invalid: {sport}")
                    stats.errors.append({"sport": sport, "error": "422", "detail": exc_str[:100]})
                    break

                elif status_code and status_code >= 500:
                    # Server error - retry with backoff
                    wait_time = (2 ** attempt) + random.random()
                    log.warning("fetch_scheduler: %d for %s, retrying in %.1fs",
                                status_code, sport, wait_time)
                    time.sleep(wait_time)
                    record_failure("odds_api", f"{status_code}: {sport}")
                    continue

                else:
                    # Unknown error
                    log.warning("fetch_scheduler: error for %s: %s", sport, exc_str[:100])
                    record_failure("odds_api", exc_str[:100])
                    stats.errors.append({"sport": sport, "error": "unknown", "detail": exc_str[:100]})
                    break

        if not success:
            # Fallback to cached data
            cached = cache.get_json(f"odds:{sport}:{regions}:{markets}")
            if cached and isinstance(cached, list):
                for e in cached:
                    raw_events.append((sport, e))
                stats.raw_events_by_sport[sport] = len(cached)
                stats.total_raw_events += len(cached)
                log.info("fetch_scheduler: using cached data for %s (%d events)", sport, len(cached))

    # Persist cycle stats
    cache.set_json(FETCH_STATS_KEY, stats.to_dict(), ttl_seconds=3600)

    log.info(
        "fetch_scheduler: fetched %d keys, %d events, skipped %d outright + %d cooldown, status=%s",
        len(stats.fetched_keys), stats.total_raw_events,
        len(stats.skipped_outright), len(stats.skipped_cooldown),
        stats.status_counts,
    )

    return raw_events, stats


def _extract_status_code(error_str: str) -> Optional[int]:
    """Try to extract HTTP status code from an exception message."""
    import re
    m = re.search(r'\b(4\d{2}|5\d{2})\b', error_str)
    if m:
        return int(m.group(1))
    return None


def _extract_retry_after(error_str: str) -> Optional[float]:
    """Try to extract Retry-After seconds from an error string."""
    import re
    m = re.search(r'retry.?after[:\s]+(\d+)', error_str, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# ---------------------------------------------------------------------------
# In-Play Terminator: per-sport adaptive fetch interval
# ---------------------------------------------------------------------------

def get_sport_fetch_interval(sport_key: str) -> int:
    """Return the optimal polling interval for a sport based on its kickoff times.

    This is the "In-Play Terminator": once ALL matches of a sport have
    kicked off (delta <= 0), the sport enters deep-sleep until the next
    trading window.  This prevents the ghost-polling bug where negative
    deltas (e.g. -600s for 10 min after kickoff) still satisfy
    ``delta <= 900`` and trigger 10s sniper polling on live games.

    Returns interval in seconds:
        No future kickoffs today:   sleep until trading window end (min 3600s)
        > 6h before nearest KO:     3600s
        1-6h before nearest KO:      600s
        15min - 1h before nearest:    60s
        < 15min before nearest:       10s  (Sniper Zone — strictly future only)
    """
    kickoffs = cache.get_json(f"kickoffs:sport:{sport_key}") or []
    if not kickoffs:
        return 3600  # No data — idle

    now = datetime.now(timezone.utc)
    future_kickoffs = []

    for ts in kickoffs:
        try:
            kt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            # CRITICAL: Only include matches that have NOT yet started.
            # If delta <= 0, the match is LIVE or FINISHED — no pre-match edge.
            if (kt - now).total_seconds() > 0:
                future_kickoffs.append(kt)
        except (ValueError, TypeError):
            continue

    # All matches already live or finished → deep sleep until next window
    if not future_kickoffs:
        window_end = get_trading_window_end()
        sleep_seconds = (window_end - now).total_seconds()
        return max(3600, int(sleep_seconds))

    delta_seconds = (min(future_kickoffs) - now).total_seconds()

    # HFT Sniper Matrix (only fires for strictly future matches)
    if delta_seconds <= 900:        # < 15 min
        return 10
    elif delta_seconds <= 3600:     # < 1 hr
        return 60
    elif delta_seconds <= 21600:    # < 6 hrs
        return 600
    else:                           # > 6 hrs
        return 3600
