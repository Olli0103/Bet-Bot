"""Dynamic bankroll management based on PlacedBet PnL history."""
from __future__ import annotations

from sqlalchemy import select

from src.core.settings import settings
from src.data.models import PlacedBet
from src.data.postgres import SessionLocal


class BankrollManager:
    def __init__(self, initial: float | None = None, owner_chat_id: str = ""):
        self._initial = initial if initial is not None else settings.initial_bankroll
        self._owner = owner_chat_id

    def get_current_bankroll(self) -> float:
        """Calculate current bankroll: initial + sum of all settled LIVE PnL.

        Excludes historical imports and paper-only signals — only real
        trading activity affects the bankroll.
        When owner_chat_id is set, only that owner's bets are considered.
        """
        with SessionLocal() as db:
            query = select(PlacedBet).where(
                PlacedBet.status.in_(["won", "lost"]),
                PlacedBet.is_training_data.is_(False),
            )
            if self._owner:
                query = query.where(PlacedBet.owner_chat_id == self._owner)
            settled = db.scalars(query).all()
        total_pnl = sum(float(b.pnl or 0.0) for b in settled)
        return max(0.0, self._initial + total_pnl)

    def get_kelly_bankroll(self) -> float:
        """Conservative bankroll for Kelly sizing (80% of current)."""
        return self.get_current_bankroll() * 0.8
