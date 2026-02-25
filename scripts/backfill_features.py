#!/usr/bin/env python3
from __future__ import annotations

import argparse
from math import isfinite

from sqlalchemy import select

from src.data.models import PlacedBet
from src.data.postgres import SessionLocal


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def backfill(limit: int = 0, only_missing: bool = True) -> dict:
    updated = 0
    scanned = 0

    with SessionLocal() as db:
        stmt = select(PlacedBet)
        rows = db.scalars(stmt).all()

        for b in rows:
            if limit and scanned >= limit:
                break
            scanned += 1

            if only_missing:
                if (b.sharp_implied_prob or 0.0) > 0 and (b.sentiment_delta or 0.0) != 0 and (b.injury_delta or 0.0) != 0:
                    continue

            odds = float(b.odds or 0.0)
            if not isfinite(odds) or odds <= 1.0:
                sharp = 0.5
            else:
                sharp = _clip(1.0 / odds, 0.01, 0.99)

            # conservative backfill: keep synthetic context features neutral
            b.sharp_implied_prob = sharp
            if b.sentiment_delta is None:
                b.sentiment_delta = 0.0
            if b.injury_delta is None:
                b.injury_delta = 0.0
            updated += 1

        db.commit()

    return {"scanned": scanned, "updated": updated}


def main():
    ap = argparse.ArgumentParser(description="Backfill ML features for historical placed_bets")
    ap.add_argument("--limit", type=int, default=0, help="max rows to process (0=all)")
    ap.add_argument("--all", action="store_true", help="rewrite all rows, not only missing")
    args = ap.parse_args()

    result = backfill(limit=args.limit, only_missing=not args.all)
    print(result)


if __name__ == "__main__":
    main()
