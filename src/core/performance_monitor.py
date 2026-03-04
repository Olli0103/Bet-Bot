"""Performance monitoring and circuit breakers for the betting pipeline.

Tracks ROI, hit rate, calibration, and Brier score over rolling windows.
Circuit breakers halt or reduce betting during losing streaks, daily loss
limits, or detected model degradation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from sqlalchemy import select

from src.core.settings import settings
from src.data.models import PlacedBet
from src.data.postgres import SessionLocal

log = logging.getLogger(__name__)


class PerformanceMonitor:
    """Tracks betting performance and provides circuit breaker signals."""

    def get_recent_performance(self, days: int = 7) -> Dict:
        """ROI, hit rate, and PnL for the last N days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with SessionLocal() as db:
            bets = db.scalars(
                select(PlacedBet).where(
                    PlacedBet.status.in_(["won", "lost"]),
                )
            ).all()

        # Filter by created_at if available, otherwise use all
        recent = []
        for b in bets:
            created = getattr(b, "created_at", None)
            if created and hasattr(created, "astimezone"):
                if created.astimezone(timezone.utc) < cutoff:
                    continue
            recent.append(b)

        if not recent:
            return {"roi": 0.0, "hit_rate": 0.0, "total_pnl": 0.0, "bets": 0, "won": 0, "lost": 0}

        won = sum(1 for b in recent if b.status == "won")
        lost = sum(1 for b in recent if b.status == "lost")
        total = won + lost
        pnl = sum(float(b.pnl or 0) for b in recent)
        staked = sum(float(b.stake or 0) for b in recent)

        return {
            "roi": round(pnl / max(1.0, staked), 4),
            "hit_rate": round(won / max(1, total), 4),
            "total_pnl": round(pnl, 2),
            "bets": total,
            "won": won,
            "lost": lost,
            "staked": round(staked, 2),
        }

    def check_circuit_breakers(self) -> Dict[str, bool]:
        """Check all circuit breaker conditions. True = breaker tripped."""
        return {
            "losing_streak": self._check_losing_streak(threshold=settings.losing_streak_threshold),
            "daily_loss_limit": self._check_daily_loss(max_pct=settings.daily_loss_limit_pct),
            "model_degradation": self._check_model_degradation(),
            "drawdown": self._check_drawdown(max_pct=settings.drawdown_max_pct, lookback_days=settings.drawdown_lookback_days),
        }

    def get_adjustment_factors(self) -> Dict[str, float]:
        """Return stake/threshold adjustments based on recent performance."""
        perf = self.get_recent_performance(days=7)
        breakers = self.check_circuit_breakers()

        # If any breaker is tripped, reduce aggressiveness
        if breakers.get("losing_streak") or breakers.get("daily_loss_limit"):
            return {"kelly_multiplier": 0.5, "min_ev": settings.min_ev_losing_streak, "active": True}

        if breakers.get("drawdown"):
            return {"kelly_multiplier": 0.5, "min_ev": settings.min_ev_drawdown, "active": True}

        if breakers.get("model_degradation"):
            return {"kelly_multiplier": 0.7, "min_ev": settings.min_ev_degradation, "active": True}

        roi = perf.get("roi", 0.0)
        if roi > 0.05:
            # Good run: slightly more aggressive
            return {"kelly_multiplier": 1.2, "min_ev": settings.min_ev_good_run, "active": False}

        # Normal
        return {"kelly_multiplier": 1.0, "min_ev": settings.min_ev_default, "active": False}

    def _check_losing_streak(self, threshold: int = 7) -> bool:
        """Return True if the last N consecutive bets are all losses."""
        with SessionLocal() as db:
            bets = db.scalars(
                select(PlacedBet)
                .where(PlacedBet.status.in_(["won", "lost"]))
                .order_by(PlacedBet.id.desc())
                .limit(threshold)
            ).all()

        if len(bets) < threshold:
            return False
        return all(b.status == "lost" for b in bets)

    def _check_daily_loss(self, max_pct: float = 0.05) -> bool:
        """Return True if today's losses exceed max_pct of bankroll."""
        today = datetime.now(timezone.utc).date()
        with SessionLocal() as db:
            bets = db.scalars(
                select(PlacedBet).where(PlacedBet.status.in_(["won", "lost"]))
            ).all()

        daily_pnl = 0.0
        for b in bets:
            created = getattr(b, "created_at", None)
            if created and hasattr(created, "date"):
                if created.date() == today:
                    daily_pnl += float(b.pnl or 0)

        bankroll = settings.initial_bankroll
        return daily_pnl < -(bankroll * max_pct)

    def _check_model_degradation(self) -> bool:
        """Return True if recent hit rate is significantly below expected."""
        perf = self.get_recent_performance(days=14)
        if perf["bets"] < 20:
            return False
        # If hit rate is below 40% over 14 days, model might be degraded
        return perf["hit_rate"] < 0.40

    def _check_drawdown(self, max_pct: float = 0.10, lookback_days: int = 7) -> bool:
        """Return True if rolling PnL drawdown exceeds max_pct of bankroll.

        Protects against multi-day losing runs that individual circuit
        breakers (streak, daily cap) might miss — e.g. 4 consecutive days
        of -2% each would not trigger the daily -5% cap but totals -8%.
        """
        perf = self.get_recent_performance(days=lookback_days)
        if perf["bets"] < 5:
            return False
        bankroll = settings.initial_bankroll
        return perf["total_pnl"] < -(bankroll * max_pct)

    def generate_report(self) -> str:
        """Generate a performance summary for Telegram."""
        perf_7d = self.get_recent_performance(days=7)
        perf_30d = self.get_recent_performance(days=30)
        breakers = self.check_circuit_breakers()
        adjustments = self.get_adjustment_factors()

        breaker_status = []
        for name, tripped in breakers.items():
            breaker_status.append(f"{'🔴' if tripped else '🟢'} {name}")

        report = (
            "📊 Performance Report\n\n"
            f"7-Tage:\n"
            f"  ROI: {perf_7d['roi']:.2%} | Hit Rate: {perf_7d['hit_rate']:.2%}\n"
            f"  PnL: {perf_7d['total_pnl']:.2f} EUR | Bets: {perf_7d['bets']}\n\n"
            f"30-Tage:\n"
            f"  ROI: {perf_30d['roi']:.2%} | Hit Rate: {perf_30d['hit_rate']:.2%}\n"
            f"  PnL: {perf_30d['total_pnl']:.2f} EUR | Bets: {perf_30d['bets']}\n\n"
            f"Circuit Breakers:\n"
            f"  {'  '.join(breaker_status)}\n\n"
            f"Adjustments: Kelly x{adjustments['kelly_multiplier']:.1f}, min_EV={adjustments['min_ev']:.3f}"
        )
        return report
