from __future__ import annotations

from sqlalchemy import select, and_, or_

from src.data.postgres import SessionLocal
from src.data.models import PlacedBet


def learning_health(live_only: bool = True) -> dict:
    """Return learning health stats.

    Parameters
    ----------
    live_only : bool
        If True (default), exclude historical imports and paper-only signals
        from PnL/bankroll calculations. Only count bets with stake > 0
        and without 'source=historical_import' in notes.
    """
    with SessionLocal() as db:
        query = select(PlacedBet)
        if live_only:
            # Exclude historical imports and zero-stake paper signals
            query = query.where(
                and_(
                    PlacedBet.stake > 0,
                    or_(
                        PlacedBet.notes.is_(None),
                        ~PlacedBet.notes.like("%source=historical_import%"),
                    ),
                )
            )
        all_bets = db.scalars(query).all()

    total = len(all_bets)
    settled = [b for b in all_bets if b.status in {"won", "lost"}]
    open_n = sum(1 for b in all_bets if b.status == "open")
    wins = sum(1 for b in settled if b.status == "won")
    losses = sum(1 for b in settled if b.status == "lost")
    pnl = round(sum(float(b.pnl or 0.0) for b in settled), 2)
    hit = round((wins / max(1, len(settled))) * 100, 2)
    return {
        "total": total,
        "settled": len(settled),
        "open": open_n,
        "wins": wins,
        "losses": losses,
        "hit_rate_pct": hit,
        "pnl": pnl,
        "live_only": live_only,
    }


def paper_learning_health() -> dict:
    """Return learning stats for paper signals (including stake=0)."""
    with SessionLocal() as db:
        all_bets = db.scalars(
            select(PlacedBet).where(
                PlacedBet.notes.like("%source=paper_signal%")
            )
        ).all()

    total = len(all_bets)
    settled = [b for b in all_bets if b.status in {"won", "lost"}]
    open_n = sum(1 for b in all_bets if b.status == "open")
    wins = sum(1 for b in settled if b.status == "won")
    losses = sum(1 for b in settled if b.status == "lost")
    playable = sum(1 for b in all_bets if b.stake and b.stake > 0)
    paper_only = total - playable

    return {
        "total_paper": total,
        "settled": len(settled),
        "open": open_n,
        "wins": wins,
        "losses": losses,
        "hit_rate_pct": round((wins / max(1, len(settled))) * 100, 2),
        "playable": playable,
        "paper_only": paper_only,
    }
