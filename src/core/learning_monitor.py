from __future__ import annotations

from sqlalchemy import select, func

from src.data.postgres import SessionLocal
from src.data.models import PlacedBet


def learning_health(live_only: bool = True, owner_chat_id: str = "") -> dict:
    """Return learning health stats.

    Parameters
    ----------
    live_only : bool
        If True (default), exclude historical imports and paper-only signals
        from PnL/bankroll calculations.  Uses the ``is_training_data`` column.
    owner_chat_id : str
        If set, scope results to this owner's portfolio only.
    """
    with SessionLocal() as db:
        query = select(PlacedBet)
        if live_only:
            query = query.where(PlacedBet.is_training_data.is_(False))
        if owner_chat_id:
            query = query.where(PlacedBet.owner_chat_id == owner_chat_id)
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
                PlacedBet.data_source == "paper_signal",
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


def training_data_stats() -> dict:
    """Return stats for historical training data (separate from live PnL).

    Provides sport breakdown, label distribution, and coverage for the
    training data view.
    """
    with SessionLocal() as db:
        total = db.scalar(
            select(func.count()).select_from(PlacedBet).where(
                PlacedBet.is_training_data.is_(True),
            )
        ) or 0

        # Breakdown by sport
        sport_rows = db.execute(
            select(
                PlacedBet.sport,
                func.count(PlacedBet.id).label("cnt"),
                func.sum(func.cast(PlacedBet.status == "won", PlacedBet.id.type)).label("won"),
            ).where(
                PlacedBet.is_training_data.is_(True),
            ).group_by(PlacedBet.sport)
        ).all()

    sports = {}
    for row in sport_rows:
        sport = row[0] or "unknown"
        cnt = int(row[1] or 0)
        won = int(row[2] or 0)
        sports[sport] = {
            "count": cnt,
            "won": won,
            "lost": cnt - won,
            "win_pct": round(won / max(1, cnt) * 100, 1),
        }

    return {
        "total_training_rows": total,
        "sports": sports,
    }
