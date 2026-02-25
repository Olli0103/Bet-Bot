from __future__ import annotations

from sqlalchemy import select

from src.data.postgres import SessionLocal
from src.data.models import PlacedBet


def learning_health() -> dict:
    with SessionLocal() as db:
        all_bets = db.scalars(select(PlacedBet)).all()
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
    }
