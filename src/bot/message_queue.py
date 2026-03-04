"""Redis-backed message queue for Core <-> Telegram worker decoupling.

Two queues:
  outbox  (Core -> Telegram)  : signals, alerts, reports to send
  inbox   (Telegram -> Core)  : user actions (mark-as-placed, deep-dive, etc.)

Messages are JSON dicts pushed via LPUSH / consumed via BRPOP.
Each message carries a dedup_key to prevent double-sends on restart.
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
INBOX_KEY = "queue:inbox"
DEDUP_PREFIX = "queue:dedup:"
DEDUP_TTL = 3600  # 1 hour dedup window


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

    Parameters
    ----------
    msg_type : str
        Message type: "text", "photo", "signal_push", "combo_push",
        "agent_alert", "report", "health".
    payload : dict
        Message content. Must include "text" for text messages.
    target : str
        "broadcast" | "primary" — routing hint for telegram worker.
    chat_ids : list of str, optional
        Explicit chat IDs. Overrides target routing.

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

    # Atomic dedup: SET NX returns True only for the first caller,
    # eliminating the race window of the old exists() + lpush() pattern.
    r = cache.client
    if not r.set(dedup_redis_key, "1", ex=DEDUP_TTL, nx=True):
        return False

    r.lpush(OUTBOX_KEY, json.dumps(msg, default=str))
    return True


def pop_outbox(timeout: int = 5) -> Optional[Dict[str, Any]]:
    """Blocking pop from outbox. Returns None on timeout."""
    r = cache.client
    result = r.brpop(OUTBOX_KEY, timeout=timeout)
    if result:
        _, raw = result
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("Invalid outbox message: %s", raw)
    return None


def outbox_len() -> int:
    r = cache.client
    return r.llen(OUTBOX_KEY)


# ── Inbox (Telegram -> Core) ──────────────────────────────────

def push_inbox(
    action: str,
    payload: Dict[str, Any],
    chat_id: str = "",
) -> None:
    """Push a user action to the inbox queue.

    Parameters
    ----------
    action : str
        "mark_placed", "deep_dive", "ghost_bet", "settings_change", etc.
    payload : dict
        Action-specific data.
    chat_id : str
        The originating chat ID.
    """
    msg = {
        "action": action,
        "payload": payload,
        "chat_id": chat_id,
        "ts": time.time(),
    }
    r = cache.client
    r.lpush(INBOX_KEY, json.dumps(msg, default=str))


def pop_inbox(timeout: int = 1) -> Optional[Dict[str, Any]]:
    """Non-blocking (short timeout) pop from inbox."""
    r = cache.client
    result = r.brpop(INBOX_KEY, timeout=timeout)
    if result:
        _, raw = result
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("Invalid inbox message: %s", raw)
    return None


def inbox_len() -> int:
    r = cache.client
    return r.llen(INBOX_KEY)
