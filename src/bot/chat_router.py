"""Multi-chat-ID routing with primary/broadcast/access control.

ENV semantics:
  TELEGRAM_CHAT_ID   = primary chat (always included)
  TELEGRAM_CHAT_IDS  = CSV of additional allowed chat IDs

Routing rules:
  - broadcast events -> all allowed IDs
  - primary-only events -> only primary ID
  - incoming messages only accepted from allowed IDs
"""
from __future__ import annotations

import logging
from typing import List, Set

from src.core.settings import settings

log = logging.getLogger(__name__)


class ChatRouter:
    """Resolves and routes messages to/from authorized Telegram chat IDs."""

    def __init__(self) -> None:
        self._primary: str = (settings.telegram_chat_id or "").strip()
        self._allowed: Set[str] = self._parse_allowed()

    def _parse_allowed(self) -> Set[str]:
        ids: Set[str] = set()
        if self._primary:
            ids.add(self._primary)
        raw = (settings.telegram_chat_ids or "").strip()
        if raw:
            for chunk in raw.split(","):
                cid = chunk.strip()
                if cid:
                    ids.add(cid)
        return ids

    # ── Public API ────────────────────────────────────────────────

    @property
    def primary_id(self) -> str:
        """The primary chat ID (for important / primary-only events)."""
        return self._primary

    @property
    def all_ids(self) -> List[str]:
        """All authorized chat IDs (for broadcast events)."""
        return sorted(self._allowed) if self._allowed else []

    def is_authorized(self, chat_id: str | int) -> bool:
        """Check if an incoming chat_id is in the allowed list."""
        return str(chat_id) in self._allowed

    def broadcast_ids(self) -> List[str]:
        """IDs for broadcast-type events (daily push, agent alerts)."""
        return self.all_ids

    def primary_only_ids(self) -> List[str]:
        """IDs for primary-only events (diagnostics, health checks)."""
        return [self._primary] if self._primary else []


# Module-level singleton
chat_router = ChatRouter()
