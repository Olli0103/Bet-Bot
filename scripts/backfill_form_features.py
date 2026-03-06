#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
import re

from sqlalchemy import select

from src.data.models import PlacedBet
from src.data.postgres import SessionLocal


def normalize_tennis_name(raw_name: str) -> str:
    """Normalize tennis player aliases to a stable short key.

    Examples:
      "Federer R." -> "federer_r"
      "R. Federer" -> "federer_r"
      "Roger Federer" -> "federer_r"
    """
    if not raw_name:
        return ""

    name = str(raw_name).lower().strip()
    name = name.replace(".", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return ""

    parts = [p for p in name.split(" ") if p]
    if len(parts) == 1:
        return parts[0]

    initials = [p for p in parts if len(p) == 1]
    long_parts = [p for p in parts if len(p) > 1]

    if initials and long_parts:
        return f"{long_parts[-1]}_{initials[0]}"

    # fallback: surname + first-name initial
    return f"{parts[-1]}_{parts[0][0]}"


def backfill_form_l5() -> dict:
    updated = 0
    with SessionLocal() as db:
        rows = db.scalars(select(PlacedBet).order_by(PlacedBet.created_at.asc(), PlacedBet.id.asc())).all()

        history = defaultdict(lambda: deque(maxlen=5))  # key=(sport, selection_norm) -> recent outcomes

        for b in rows:
            sport = str(b.sport)
            sel = str(b.selection)
            if sport.startswith("tennis"):
                sel = normalize_tennis_name(sel)
            key = (sport, sel)
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
