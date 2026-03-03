"""Structured observability module for the Bet-Bot pipeline.

Provides:
  - Correlation / cycle IDs for tracing requests across components
  - Domain-specific structured loggers (fetch, enrichment, agent, signal,
    telegram, db, training)
  - Heartbeat summaries (single compact health line per cycle)
  - Duration tracking helpers

All loggers emit human-readable messages while preserving machine-parseable
key=value fields for downstream analysis.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Correlation / Cycle ID
# ---------------------------------------------------------------------------

_current_cycle_id: Optional[str] = None


def new_cycle_id() -> str:
    """Generate and set a new global cycle ID (short UUID)."""
    global _current_cycle_id
    _current_cycle_id = uuid.uuid4().hex[:12]
    return _current_cycle_id


def get_cycle_id() -> str:
    """Return the current cycle ID, creating one if none exists."""
    global _current_cycle_id
    if _current_cycle_id is None:
        _current_cycle_id = uuid.uuid4().hex[:12]
    return _current_cycle_id


# ---------------------------------------------------------------------------
# Duration tracking
# ---------------------------------------------------------------------------

@contextmanager
def track_duration(operation: str, logger: Optional[logging.Logger] = None):
    """Context manager that logs operation duration.

    Usage::

        with track_duration("fetch_odds", log):
            data = fetcher.get_odds()
    """
    _log = logger or log
    cid = get_cycle_id()
    start = time.monotonic()
    _log.info("op=%s cycle=%s status=start", operation, cid)
    try:
        yield
    except Exception as exc:
        elapsed = time.monotonic() - start
        _log.error(
            "op=%s cycle=%s status=error duration=%.2fs error=%s",
            operation, cid, elapsed, type(exc).__name__,
        )
        raise
    else:
        elapsed = time.monotonic() - start
        _log.info(
            "op=%s cycle=%s status=done duration=%.2fs",
            operation, cid, elapsed,
        )


# ---------------------------------------------------------------------------
# Domain-specific structured loggers
# ---------------------------------------------------------------------------

class StructuredLogger:
    """Logger wrapper that auto-injects cycle_id and domain tag."""

    def __init__(self, domain: str):
        self.domain = domain
        self._log = logging.getLogger(f"betbot.{domain}")

    def _fmt(self, msg: str, **kv: Any) -> str:
        cid = get_cycle_id()
        parts = [f"[{self.domain}]", f"cycle={cid}", msg]
        for k, v in kv.items():
            parts.append(f"{k}={v}")
        return " ".join(parts)

    def info(self, msg: str, **kv: Any) -> None:
        self._log.info(self._fmt(msg, **kv))

    def warning(self, msg: str, **kv: Any) -> None:
        self._log.warning(self._fmt(msg, **kv))

    def error(self, msg: str, **kv: Any) -> None:
        self._log.error(self._fmt(msg, **kv))

    def debug(self, msg: str, **kv: Any) -> None:
        self._log.debug(self._fmt(msg, **kv))


# Pre-built domain loggers
fetch_log = StructuredLogger("fetch")
enrichment_log = StructuredLogger("enrichment")
agent_log = StructuredLogger("agent")
signal_log = StructuredLogger("signal")
telegram_log = StructuredLogger("telegram")
db_log = StructuredLogger("db")
training_log = StructuredLogger("training")


# ---------------------------------------------------------------------------
# Lifecycle event helpers
# ---------------------------------------------------------------------------

def log_fetch_lifecycle(
    source: str,
    sport: str,
    events: int,
    duration_s: float,
    status: str = "ok",
) -> None:
    """Log a fetch lifecycle event (start/end/duration/result counts)."""
    fetch_log.info(
        "fetch_complete",
        source=source,
        sport=sport,
        events=events,
        duration=f"{duration_s:.2f}s",
        status=status,
    )


def log_enrichment_lifecycle(
    provider: str,
    status: str,
    retries: int = 0,
    timeout: bool = False,
    detail: str = "",
) -> None:
    """Log an enrichment provider lifecycle event."""
    enrichment_log.info(
        "enrichment_result",
        provider=provider,
        status=status,
        retries=retries,
        timeout=timeout,
        detail=detail,
    )


def log_agent_cycle(
    alerts: int,
    bets_placed: int,
    bets_skipped: int,
    duration_s: float,
    mode: str = "NORMAL",
) -> None:
    """Log an agent orchestration cycle summary."""
    agent_log.info(
        "cycle_complete",
        mode=mode,
        alerts=alerts,
        placed=bets_placed,
        skipped=bets_skipped,
        duration=f"{duration_s:.2f}s",
    )


def log_signal_pipeline(
    events_in: int,
    signals_out: int,
    combos_generated: int,
    combos_filtered: int,
    filter_reasons: Optional[Dict[str, int]] = None,
) -> None:
    """Log signal pipeline throughput."""
    signal_log.info(
        "pipeline_complete",
        events_in=events_in,
        signals_out=signals_out,
        combos_gen=combos_generated,
        combos_filtered=combos_filtered,
        filter_reasons=filter_reasons or {},
    )


def log_telegram_io(
    action: str,
    success: bool,
    latency_ms: Optional[float] = None,
    detail: str = "",
) -> None:
    """Log Telegram I/O events."""
    telegram_log.info(
        action,
        success=success,
        latency_ms=f"{latency_ms:.0f}" if latency_ms else "n/a",
        detail=detail,
    )


def log_db_write(
    operation: str,
    count: int,
    errors: int = 0,
    detail: str = "",
) -> None:
    """Log database write operations."""
    db_log.info(
        operation,
        count=count,
        errors=errors,
        detail=detail,
    )


def log_training_run(
    sport: str,
    samples: int,
    brier: Optional[float] = None,
    skipped: bool = False,
    reason: str = "",
) -> None:
    """Log a training run result."""
    training_log.info(
        "training_result",
        sport=sport,
        samples=samples,
        brier=f"{brier:.6f}" if brier is not None else "n/a",
        skipped=skipped,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Heartbeat summary
# ---------------------------------------------------------------------------

def heartbeat_summary(
    fetch_ok: bool = True,
    enrichment_ok: bool = True,
    agent_ok: bool = True,
    db_ok: bool = True,
    telegram_ok: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a single compact heartbeat line for the current cycle.

    Example output::

        HEARTBEAT cycle=a1b2c3d4e5f6 ts=2026-03-03T14:00:00Z
        fetch=OK enrich=OK agent=OK db=OK tg=OK
    """
    cid = get_cycle_id()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _s(ok: bool) -> str:
        return "OK" if ok else "DEGRADED"

    line = (
        f"HEARTBEAT cycle={cid} ts={ts} "
        f"fetch={_s(fetch_ok)} enrich={_s(enrichment_ok)} "
        f"agent={_s(agent_ok)} db={_s(db_ok)} tg={_s(telegram_ok)}"
    )

    if extra:
        extras = " ".join(f"{k}={v}" for k, v in extra.items())
        line += f" {extras}"

    log.info(line)
    return line
