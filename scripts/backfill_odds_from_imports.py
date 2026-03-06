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


def _calc_overround(prices: dict) -> float | None:
    vals = []
    for v in prices.values():
        try:
            vf = float(v)
        except Exception:
            continue
        if vf > 1.0:
            vals.append(vf)
    if len(vals) < 2:
        return None
    return round(max(0.001, sum(1.0 / x for x in vals) - 1.0), 4)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    return None


def backfill_football(folder: Path) -> int:
    updates = 0
    with SessionLocal() as db:
        rows = db.scalars(select(PlacedBet).where(PlacedBet.sport.like("soccer_%"))).all()
        idx = {(r.event_id, r.selection): r for r in rows}

        for p in sorted(folder.glob("*.csv")):
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue
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

                # close proxies (market consensus) — prefer Pinnacle closing first
                oh_close = _first_num(r, ["PSCH", "PSH", "AvgH", "MaxH", "B365H"])
                od_close = _first_num(r, ["PSCD", "PSD", "AvgD", "MaxD", "B365D"])
                oa_close = _first_num(r, ["PSCA", "PSA", "AvgA", "MaxA", "B365A"])

                sharp_prices_h2h = {"home": oh_close, "draw": od_close, "away": oa_close}
                sharp_vig_h2h = _calc_overround(sharp_prices_h2h)

                for selection, o_open, o_close in [(home, oh_open, oh_close), ("Draw", od_open, od_close), (away, oa_open, oa_close)]:
                    key = (event_id, selection)
                    bet = idx.get(key)
                    if not bet:
                        continue
                    if o_open and o_close:
                        bet.odds_open = float(o_open)
                        bet.odds_close = float(o_close)
                        bet.clv = (float(bet.odds) / float(o_close)) - 1.0 if float(o_close) > 1.0 else 0.0
                        meta = dict(bet.meta_features or {})
                        if sharp_vig_h2h is not None:
                            meta["sharp_prices_h2h"] = {k: float(v) for k, v in sharp_prices_h2h.items() if v and v > 1.0}
                            meta["sharp_vig_true"] = sharp_vig_h2h
                            meta["sharp_vig_method"] = "book_overround_1x2"
                            bet.sharp_vig = sharp_vig_h2h
                        bet.meta_features = meta
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

                sharp_prices_h2h = {"home": w_close, "away": l_close}
                sharp_vig_h2h = _calc_overround(sharp_prices_h2h)

                for selection, o_open, o_close in [(winner, w_open, w_close), (loser, l_open, l_close)]:
                    key = (event_id, selection)
                    bet = idx.get(key)
                    if not bet:
                        continue
                    if o_open and o_close:
                        bet.odds_open = float(o_open)
                        bet.odds_close = float(o_close)
                        bet.clv = (float(bet.odds) / float(o_close)) - 1.0 if float(o_close) > 1.0 else 0.0
                        meta = dict(bet.meta_features or {})
                        if sharp_vig_h2h is not None:
                            meta["sharp_prices_h2h"] = {k: float(v) for k, v in sharp_prices_h2h.items() if v and v > 1.0}
                            meta["sharp_vig_true"] = sharp_vig_h2h
                            meta["sharp_vig_method"] = "book_overround_2way"
                            bet.sharp_vig = sharp_vig_h2h
                        bet.meta_features = meta
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
