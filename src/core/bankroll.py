"""Dynamic bankroll management based on PlacedBet PnL history.

Uses SQL-level aggregation (func.sum) instead of loading all rows into
Python to avoid OOM on large bet histories.
"""
from __future__ import annotations

from sqlalchemy import func, select

from src.core.settings import settings
from src.data.models import PlacedBet
from src.data.postgres import SessionLocal


class BankrollManager:
    def __init__(self, initial: float | None = None, owner_chat_id: str = ""):
        self._initial = initial if initial is not None else settings.initial_bankroll
        self._owner = owner_chat_id

    def get_current_bankroll(self) -> float:
        """Calculate current bankroll: initial + sum of all settled LIVE PnL.

        Uses SQL SUM() aggregation — never loads individual rows into memory.
        Excludes historical imports and paper-only signals.
        When owner_chat_id is set, only that owner's bets are considered.
        """
        with SessionLocal() as db:
            query = select(func.coalesce(func.sum(PlacedBet.pnl), 0.0)).where(
                PlacedBet.status.in_(["won", "lost"]),
                PlacedBet.is_training_data.is_(False),
            )
            if self._owner:
                query = query.where(PlacedBet.owner_chat_id == self._owner)
            total_pnl = float(db.scalar(query) or 0.0)
        return max(0.0, self._initial + total_pnl)

    def get_kelly_bankroll(self) -> float:
        """Conservative bankroll for Kelly sizing (80% of current)."""
        return self.get_current_bankroll() * 0.8
