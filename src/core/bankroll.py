"""Dynamic bankroll management based on PlacedBet PnL history.

Uses SQL-level aggregation (func.sum) instead of loading all rows into
Python to avoid OOM on large bet histories.

LUGAS/OASIS Compliance (GlüStV 2021 §6c):
    German online gambling regulations impose a EUR 1,000 monthly deposit
    limit across all bookmakers.  The ``get_lugas_remaining`` method
    simulates this constraint within the DSS's own bankroll logic so
    the system never recommends stakes that would violate the operator's
    actual account limits.
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import func, select, extract

from src.core.settings import settings
from src.data.models import PlacedBet
from src.data.postgres import SessionLocal
from src.data.redis_cache import cache

import logging

log = logging.getLogger(__name__)

# Redis key tracking total pending (unsettled) exposure in EUR
PENDING_EXPOSURE_KEY = "bankroll:pending_exposure"
PENDING_EXPOSURE_TTL = 24 * 3600  # 24 hours — resets daily

# GlüStV 2021 §6c: Monthly deposit limit for regulated German bookmakers
LUGAS_MONTHLY_DEPOSIT_LIMIT = 1000.0  # EUR


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

    def get_free_margin(self) -> float:
        """Return bankroll minus pending (unsettled) exposure.

        This is the correct base for Kelly sizing when multiple bets
        are placed simultaneously.  Without this, 10 concurrent signals
        each computing Kelly on the full bankroll leads to catastrophic
        over-staking (e.g. 50% of bankroll at risk instead of 5%).
        """
        bankroll = self.get_current_bankroll()
        pending = self.get_pending_exposure()
        free = max(0.0, bankroll - pending)
        return free

    def get_pending_exposure(self) -> float:
        """Return total EUR currently locked in unsettled bets."""
        raw = cache.get_json(PENDING_EXPOSURE_KEY)
        if raw is not None:
            return max(0.0, float(raw))
        # Fallback: query DB for pending bets
        try:
            with SessionLocal() as db:
                query = select(func.coalesce(func.sum(PlacedBet.stake), 0.0)).where(
                    PlacedBet.status == "pending",
                    PlacedBet.is_training_data.is_(False),
                )
                if self._owner:
                    query = query.where(PlacedBet.owner_chat_id == self._owner)
                pending = float(db.scalar(query) or 0.0)
            return max(0.0, pending)
        except Exception:
            return 0.0

    def add_pending_exposure(self, stake: float) -> float:
        """Atomically add a new bet's stake to pending exposure.

        Returns the new total pending exposure.
        """
        current = self.get_pending_exposure()
        new_total = current + stake
        cache.set_json(PENDING_EXPOSURE_KEY, new_total, ttl_seconds=PENDING_EXPOSURE_TTL)
        log.info(
            "Pending exposure updated: %.2f + %.2f = %.2f",
            current, stake, new_total,
        )
        return new_total

    def release_pending_exposure(self, stake: float) -> float:
        """Remove a settled bet's stake from pending exposure.

        Returns the new total pending exposure.
        """
        current = self.get_pending_exposure()
        new_total = max(0.0, current - stake)
        cache.set_json(PENDING_EXPOSURE_KEY, new_total, ttl_seconds=PENDING_EXPOSURE_TTL)
        return new_total

    def get_monthly_staked(self) -> float:
        """Return total EUR staked this calendar month (LUGAS simulation).

        Queries the PlacedBet table for all stakes placed in the current
        month, regardless of outcome.  This simulates the monthly deposit
        tracking that LUGAS performs across all regulated bookmakers.
        """
        now = datetime.now(timezone.utc)
        try:
            with SessionLocal() as db:
                query = select(func.coalesce(func.sum(PlacedBet.stake), 0.0)).where(
                    PlacedBet.is_training_data.is_(False),
                    extract("year", PlacedBet.created_at) == now.year,
                    extract("month", PlacedBet.created_at) == now.month,
                )
                if self._owner:
                    query = query.where(PlacedBet.owner_chat_id == self._owner)
                return max(0.0, float(db.scalar(query) or 0.0))
        except Exception:
            return 0.0

    def get_lugas_remaining(self) -> float:
        """Return remaining EUR available under the LUGAS monthly limit.

        The system will never recommend a stake larger than this value,
        preventing the operator from exceeding the GlüStV §6c deposit cap.
        """
        staked = self.get_monthly_staked()
        remaining = max(0.0, LUGAS_MONTHLY_DEPOSIT_LIMIT - staked)
        if remaining < 50.0:
            log.warning(
                "LUGAS limit approaching: %.2f/%.2f EUR staked this month",
                staked, LUGAS_MONTHLY_DEPOSIT_LIMIT,
            )
        return remaining

    def get_dss_safe_margin(self) -> float:
        """Return the bankroll available for Kelly sizing, capped by LUGAS.

        This is the correct base for DSS recommendations:
        ``min(free_margin, lugas_remaining)``  — the operator cannot
        stake more than either their bankroll allows or LUGAS permits.
        """
        free = self.get_free_margin()
        lugas = self.get_lugas_remaining()
        return min(free, lugas)
