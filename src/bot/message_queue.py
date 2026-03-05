"""Redis-backed message queue for Core <-> Telegram worker decoupling.

Two queues:
  outbox  (Core -> Telegram)  : signals, alerts, reports to send
  inbox   (Telegram -> Core)  : user actions (mark-as-placed, deep-dive, etc.)

Reliable Queue Pattern:
  Uses RPOPLPUSH (BRPOPLPUSH) to atomically move messages from the main
  queue to a processing queue.  After successful delivery, the message
  is removed from the processing queue via LREM.  If the consumer crashes
  before acknowledging, the message remains in the processing queue and
  can be recovered on restart via ``recover_unacked()``.

  This prevents the data-loss scenario of plain BRPOP where a pop + crash
  loses the message forever.
"""
from __future__ import annotations

import json
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional

from src.data.redis_cache import cache

log = logging.getLogger(__name__)

OUTBOX_KEY = "queue:outbox"
OUTBOX_PROCESSING_KEY = "queue:outbox:processing"
INBOX_KEY = "queue:inbox"
INBOX_PROCESSING_KEY = "queue:inbox:processing"
DEDUP_PREFIX = "queue:dedup:"
DEDUP_TTL = 3600  # 1 hour dedup window

# Messages older than this in the processing queue are considered dead
_PROCESSING_TTL = 300  # 5 minutes


def _dedup_key(msg: Dict[str, Any]) -> str:
    """Deterministic hash for dedup."""
    raw = json.dumps(msg, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Outbox (Core -> Telegram) ──────────────────────────────────

def push_outbox(
    msg_type: str,
    payload: Dict[str, Any],
    target: str = "broadcast",
    chat_ids: Optional[List[str]] = None,
) -> bool:
    """Push a message to the outbox queue.

    Returns True if enqueued, False if dedup hit.
    """
    msg = {
        "type": msg_type,
        "payload": payload,
        "target": target,
        "chat_ids": chat_ids,
        "ts": time.time(),
    }

    dk = _dedup_key(msg)
    dedup_redis_key = f"{DEDUP_PREFIX}{dk}"

    r = cache.client
    if not r.set(dedup_redis_key, "1", ex=DEDUP_TTL, nx=True):
        return False

    r.lpush(OUTBOX_KEY, json.dumps(msg, default=str))
    return True


def pop_outbox(timeout: int = 5) -> Optional[Dict[str, Any]]:
    """Pop from outbox using reliable queue pattern.

    The message is atomically moved to the processing queue.
    Call ``ack_outbox(raw_msg)`` after successful delivery,
    or the message will be recovered on next ``recover_unacked()``.

    Returns (parsed_dict, raw_bytes) or None on timeout.
    """
    r = cache.client
    if timeout > 0:
        raw = r.brpoplpush(OUTBOX_KEY, OUTBOX_PROCESSING_KEY, timeout=timeout)
    else:
        raw = r.rpoplpush(OUTBOX_KEY, OUTBOX_PROCESSING_KEY)
    if raw:
        try:
            parsed = json.loads(raw)
            # Embed raw bytes for ack
            parsed["_raw"] = raw if isinstance(raw, (str, bytes)) else json.dumps(parsed, default=str)
            return parsed
        except (json.JSONDecodeError, TypeError):
            log.warning("Invalid outbox message: %s", raw)
            # Remove corrupt message from processing queue
            r.lrem(OUTBOX_PROCESSING_KEY, 1, raw)
    return None


def ack_outbox(msg: Dict[str, Any]) -> None:
    """Acknowledge successful delivery — remove from processing queue."""
    r = cache.client
    raw = msg.get("_raw")
    if raw:
        r.lrem(OUTBOX_PROCESSING_KEY, 1, raw)


def outbox_len() -> int:
    r = cache.client
    return r.llen(OUTBOX_KEY)


# ── Inbox (Telegram -> Core) ──────────────────────────────────

def push_inbox(
    action: str,
    payload: Dict[str, Any],
    chat_id: str = "",
) -> None:
    """Push a user action to the inbox queue."""
    msg = {
        "action": action,
        "payload": payload,
        "chat_id": chat_id,
        "ts": time.time(),
    }
    r = cache.client
    r.lpush(INBOX_KEY, json.dumps(msg, default=str))


def pop_inbox(timeout: int = 1) -> Optional[Dict[str, Any]]:
    """Pop from inbox using reliable queue pattern."""
    r = cache.client
    if timeout > 0:
        raw = r.brpoplpush(INBOX_KEY, INBOX_PROCESSING_KEY, timeout=timeout)
    else:
        raw = r.rpoplpush(INBOX_KEY, INBOX_PROCESSING_KEY)
    if raw:
        try:
            parsed = json.loads(raw)
            parsed["_raw"] = raw if isinstance(raw, (str, bytes)) else json.dumps(parsed, default=str)
            return parsed
        except (json.JSONDecodeError, TypeError):
            log.warning("Invalid inbox message: %s", raw)
            r.lrem(INBOX_PROCESSING_KEY, 1, raw)
    return None


def ack_inbox(msg: Dict[str, Any]) -> None:
    """Acknowledge successful processing — remove from inbox processing queue."""
    r = cache.client
    raw = msg.get("_raw")
    if raw:
        r.lrem(INBOX_PROCESSING_KEY, 1, raw)


def inbox_len() -> int:
    r = cache.client
    return r.llen(INBOX_KEY)


# ── Recovery ──────────────────────────────────────────────────

def recover_unacked() -> int:
    """Move stale messages from processing queues back to main queues.

    Call this on worker startup to recover messages that were popped
    but never acked (e.g. due to a crash).

    Returns the number of recovered messages.
    """
    r = cache.client
    recovered = 0

    for proc_key, main_key in [
        (OUTBOX_PROCESSING_KEY, OUTBOX_KEY),
        (INBOX_PROCESSING_KEY, INBOX_KEY),
    ]:
        while True:
            raw = r.rpoplpush(proc_key, main_key)
            if raw is None:
                break
            recovered += 1
            log.info("Recovered unacked message from %s", proc_key)

    if recovered > 0:
        log.warning("Recovered %d unacked messages on startup", recovered)
    return recovered


def recover_stale_processing() -> int:
    """Periodically recover messages stuck in processing queues.

    Unlike ``recover_unacked()`` (called on startup), this is safe to
    call from a running worker on a timer.  It only recovers messages
    older than ``_PROCESSING_TTL`` seconds, leaving recently-popped
    messages alone (they may still be in-flight).

    Without this, a worker crash between pop and ack leaves messages
    stranded in the processing queue until the *next* full restart.
    """
    r = cache.client
    recovered = 0
    now = time.time()

    for proc_key, main_key in [
        (OUTBOX_PROCESSING_KEY, OUTBOX_KEY),
        (INBOX_PROCESSING_KEY, INBOX_KEY),
    ]:
        # Peek at all messages in the processing queue
        try:
            items = r.lrange(proc_key, 0, -1)
        except Exception:
            continue

        for raw in items:
            try:
                msg = json.loads(raw)
                ts = float(msg.get("ts", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                ts = 0  # corrupt message — recover it

            if now - ts > _PROCESSING_TTL:
                # Move back to main queue and remove from processing
                try:
                    r.lpush(main_key, raw)
                    r.lrem(proc_key, 1, raw)
                    recovered += 1
                except Exception:
                    pass

    if recovered > 0:
        log.warning("Recovered %d stale processing messages", recovered)
    return recovered
