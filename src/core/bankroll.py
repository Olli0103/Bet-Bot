"""Dynamic bankroll management based on PlacedBet PnL history."""
from __future__ import annotations

from sqlalchemy import select

from src.core.settings import settings
from src.data.models import PlacedBet
from src.data.postgres import SessionLocal


class BankrollManager:
    def __init__(self, initial: float | None = None):
        self._initial = initial if initial is not None else settings.initial_bankroll

    def get_current_bankroll(self) -> float:
        """Calculate current bankroll: initial + sum of all settled LIVE PnL.

        Excludes historical imports and paper-only signals — only real
        trading activity affects the bankroll.
        """
        with SessionLocal() as db:
            settled = db.scalars(
                select(PlacedBet).where(
                    PlacedBet.status.in_(["won", "lost"]),
                    PlacedBet.is_training_data.is_(False),
                )
            ).all()
        total_pnl = sum(float(b.pnl or 0.0) for b in settled)
        return max(0.0, self._initial + total_pnl)

    def get_kelly_bankroll(self) -> float:
        """Conservative bankroll for Kelly sizing (80% of current)."""
        return self.get_current_bankroll() * 0.8
