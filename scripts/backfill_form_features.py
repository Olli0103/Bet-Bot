#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque

from sqlalchemy import select

from src.data.models import PlacedBet
from src.data.postgres import SessionLocal


def backfill_form_l5() -> dict:
    updated = 0
    with SessionLocal() as db:
        rows = db.scalars(select(PlacedBet).order_by(PlacedBet.created_at.asc(), PlacedBet.id.asc())).all()

        history = defaultdict(lambda: deque(maxlen=5))  # key=(sport, selection) -> recent outcomes

        for b in rows:
            key = (str(b.sport), str(b.selection))
            h = history[key]

            games = len(h)
            winrate = (sum(h) / games) if games > 0 else 0.0

            b.form_games_l5 = int(games)
            b.form_winrate_l5 = float(round(winrate, 6))
            updated += 1

            if b.status == 'won':
                h.append(1)
            elif b.status == 'lost':
                h.append(0)
            # open bets do not update historical outcome trail

        db.commit()

    return {"updated": updated}


if __name__ == '__main__':
    print(backfill_form_l5())
