#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from src.data.models import PlacedBet
from src.data.postgres import SessionLocal


def _mk_event_id(parts: list[str]) -> str:
    s = "|".join([str(x or "") for x in parts])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _first_num(row, cols: list[str]) -> float | None:
    for c in cols:
        if c in row and pd.notna(row[c]):
            try:
                v = float(row[c])
                if v > 1.0:
                    return v
            except Exception:
                pass
    return None


def backfill_football(folder: Path) -> int:
    updates = 0
    with SessionLocal() as db:
        rows = db.scalars(select(PlacedBet).where(PlacedBet.sport.like("soccer_%"))).all()
        idx = {(r.event_id, r.selection): r for r in rows}

        for p in sorted(folder.glob("*.csv")):
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                home = str(r.get("HomeTeam") or "")
                away = str(r.get("AwayTeam") or "")
                if not home or not away:
                    continue
                event_id = _mk_event_id(["football", r.get("Div"), r.get("Date"), home, away])

                # open proxies (book baseline)
                oh_open = _first_num(r, ["B365H", "WHH", "VCH", "IWH", "LBH"])
                od_open = _first_num(r, ["B365D", "WHD", "VCD", "IWD", "LBD"])
                oa_open = _first_num(r, ["B365A", "WHA", "VCA", "IWA", "LBA"])

                # close proxies (market consensus)
                oh_close = _first_num(r, ["PSH", "AvgH", "MaxH", "B365H"])
                od_close = _first_num(r, ["PSD", "AvgD", "MaxD", "B365D"])
                oa_close = _first_num(r, ["PSA", "AvgA", "MaxA", "B365A"])

                for selection, o_open, o_close in [(home, oh_open, oh_close), ("Draw", od_open, od_close), (away, oa_open, oa_close)]:
                    key = (event_id, selection)
                    bet = idx.get(key)
                    if not bet:
                        continue
                    if o_open and o_close:
                        bet.odds_open = float(o_open)
                        bet.odds_close = float(o_close)
                        bet.clv = (float(bet.odds) / float(o_close)) - 1.0 if float(o_close) > 1.0 else 0.0
                        updates += 1
        db.commit()
    return updates


def backfill_tennis(folder: Path) -> int:
    updates = 0
    with SessionLocal() as db:
        rows = db.scalars(select(PlacedBet).where(PlacedBet.sport.like("tennis_%"))).all()
        idx = {(r.event_id, r.selection): r for r in rows}

        for p in sorted(folder.glob("*.xlsx")):
            df = pd.read_excel(p)
            for _, r in df.iterrows():
                winner = str(r.get("Winner") or "")
                loser = str(r.get("Loser") or "")
                if not winner or not loser:
                    continue
                event_id = _mk_event_id(["tennis", p.name, r.get("Date"), r.get("Tournament"), winner, loser])

                w_open = _first_num(r, ["B365W", "CBW", "EXW"])
                l_open = _first_num(r, ["B365L", "CBL", "EXL"])
                w_close = _first_num(r, ["PSW", "AvgW", "MaxW", "B365W"])
                l_close = _first_num(r, ["PSL", "AvgL", "MaxL", "B365L"])

                for selection, o_open, o_close in [(winner, w_open, w_close), (loser, l_open, l_close)]:
                    key = (event_id, selection)
                    bet = idx.get(key)
                    if not bet:
                        continue
                    if o_open and o_close:
                        bet.odds_open = float(o_open)
                        bet.odds_close = float(o_close)
                        bet.clv = (float(bet.odds) / float(o_close)) - 1.0 if float(o_close) > 1.0 else 0.0
                        updates += 1
        db.commit()
    return updates


def main():
    base = Path("data/imports")
    f = backfill_football(base / "football")
    t = backfill_tennis(base / "tennis")
    print({"football_updates": f, "tennis_updates": t, "total_updates": f + t})


if __name__ == "__main__":
    main()
