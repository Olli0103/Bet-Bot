"""Per-source circuit breaker and health tracking.

Tracks API call success/failure per data source (OddsAPI, TheSportsDB,
football-data.org, API-Sports, NewsAPI, Rotowire RSS). Trips a circuit
breaker when a source has too many consecutive failures, preventing
wasted API quota and cascading errors.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

HEALTH_KEY_PREFIX = "source_health:"
DEFAULT_TTL = 3600  # 1 hour

# Circuit breaker config per source
SOURCE_CONFIG: Dict[str, Dict[str, Any]] = {
    "odds_api": {"max_failures": 5, "cooldown_seconds": 300, "label": "The Odds API"},
    "thesportsdb": {"max_failures": 10, "cooldown_seconds": 180, "label": "TheSportsDB"},
    "football_data": {"max_failures": 5, "cooldown_seconds": 300, "label": "football-data.org"},
    "apisports": {"max_failures": 5, "cooldown_seconds": 300, "label": "API-Sports"},
    "newsapi": {"max_failures": 8, "cooldown_seconds": 600, "label": "NewsAPI"},
    "rotowire_rss": {"max_failures": 10, "cooldown_seconds": 180, "label": "Rotowire RSS"},
    "ollama": {"max_failures": 3, "cooldown_seconds": 120, "label": "Ollama LLM"},
    "open_meteo": {"max_failures": 10, "cooldown_seconds": 300, "label": "Open-Meteo"},
}


def _health_key(source: str) -> str:
    return f"{HEALTH_KEY_PREFIX}{source}"


def record_success(source: str) -> None:
    """Record a successful API call. Resets failure counter."""
    key = _health_key(source)
    data = cache.get_json(key) or {}
    data["failures"] = 0
    data["last_success"] = time.time()
    data["total_success"] = data.get("total_success", 0) + 1
    data["status"] = "healthy"
    cache.set_json(key, data, ttl_seconds=DEFAULT_TTL)


def record_429(source: str, cooldown_seconds: int = 0) -> None:
    """Record a 429 (rate-limit) response with automatic cooldown.

    Sets a short cooldown on the source so subsequent calls skip it,
    preventing unnecessary quota burn.
    """
    if cooldown_seconds <= 0:
        cooldown_seconds = SOURCE_CONFIG.get(source, {}).get("cooldown_seconds", 300)
    cache.set_json(
        f"source_429:{source}",
        {"until": time.time() + cooldown_seconds, "ts": time.time()},
        ttl_seconds=cooldown_seconds + 10,
    )
    log.warning("429 cooldown set for %s (%ds)", source, cooldown_seconds)
    record_failure(source, f"429 rate limit (cooldown {cooldown_seconds}s)")


def is_429_cooled(source: str) -> bool:
    """Check if source is in 429-cooldown and shouldn't be queried."""
    cd = cache.get_json(f"source_429:{source}")
    if not cd:
        return False
    return time.time() < cd.get("until", 0)


def record_failure(source: str, error: str = "") -> None:
    """Record a failed API call. May trip the circuit breaker."""
    key = _health_key(source)
    data = cache.get_json(key) or {}
    old_status = data.get("status", "healthy")
    failures = data.get("failures", 0) + 1
    data["failures"] = failures
    data["last_failure"] = time.time()
    data["last_error"] = str(error)[:200]
    data["total_failures"] = data.get("total_failures", 0) + 1

    config = SOURCE_CONFIG.get(source, {"max_failures": 5, "cooldown_seconds": 300})
    if failures >= config["max_failures"]:
        data["status"] = "open"  # circuit breaker tripped
        data["breaker_tripped_at"] = time.time()
        log.warning(
            "Circuit breaker TRIPPED for %s after %d failures: %s",
            source, failures, error,
        )
        # Alert on transition to "open" only (not every failure while open)
        if old_status != "open":
            _push_breaker_alert(source, failures, str(error)[:100], config)
    else:
        data["status"] = "degraded" if failures >= config["max_failures"] // 2 else "healthy"

    cache.set_json(key, data, ttl_seconds=DEFAULT_TTL)


def _push_breaker_alert(
    source: str, failures: int, error: str, config: dict
) -> None:
    """Push a Telegram alert when a circuit breaker trips."""
    try:
        from src.bot.message_queue import push_outbox

        label = config.get("label", source)
        cooldown = config.get("cooldown_seconds", 300)
        push_outbox("text", {
            "text": (
                f"🔴 Circuit Breaker: {label}\n"
                f"{failures} aufeinanderfolgende Fehler — Quelle pausiert ({cooldown}s Cooldown).\n"
                f"Letzter Fehler: {error}"
            ),
        }, target="primary")
    except Exception as exc:
        log.debug("Could not push breaker alert: %s", exc)


def is_available(source: str) -> bool:
    """Check if a source is available (circuit breaker not tripped).

    Returns True if the source is healthy or in cooldown-expired state.
    """
    key = _health_key(source)
    data = cache.get_json(key)
    if not data:
        return True

    status = data.get("status", "healthy")
    if status != "open":
        return True

    # Check if cooldown has expired
    config = SOURCE_CONFIG.get(source, {"cooldown_seconds": 300})
    tripped_at = data.get("breaker_tripped_at", 0)
    elapsed = time.time() - tripped_at

    if elapsed >= config["cooldown_seconds"]:
        # Half-open: allow one attempt
        data["status"] = "half_open"
        cache.set_json(key, data, ttl_seconds=DEFAULT_TTL)
        return True

    return False


def get_health_report() -> str:
    """Generate a health report for all sources."""
    lines = ["📡 Data Source Health", "━━━━━━━━━━━━━━━━━━━━"]

    for source, config in SOURCE_CONFIG.items():
        key = _health_key(source)
        data = cache.get_json(key) or {}
        status = data.get("status", "unknown")
        failures = data.get("failures", 0)
        total_ok = data.get("total_success", 0)
        total_fail = data.get("total_failures", 0)

        emoji = {
            "healthy": "🟢",
            "degraded": "🟡",
            "half_open": "🟠",
            "open": "🔴",
            "unknown": "⚪",
        }.get(status, "⚪")

        label = config.get("label", source)
        lines.append(
            f"{emoji} {label}: {status} "
            f"(ok:{total_ok} fail:{total_fail} streak:{failures})"
        )

        if status == "open":
            last_err = data.get("last_error", "")
            if last_err:
                lines.append(f"   └ {last_err[:60]}")

    return "\n".join(lines)


def get_all_health() -> Dict[str, Dict[str, Any]]:
    """Return raw health data for all sources."""
    result = {}
    for source in SOURCE_CONFIG:
        key = _health_key(source)
        data = cache.get_json(key) or {"status": "unknown", "failures": 0}
        result[source] = data
    return result
