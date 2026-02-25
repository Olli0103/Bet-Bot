#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from sqlalchemy import select, func

from src.data.models import Base, PlacedBet
from src.data.postgres import SessionLocal, engine


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


def import_football(folder: Path, limit_files: int = 0) -> int:
    files = sorted(folder.glob("*.csv"))
    if limit_files > 0:
        files = files[:limit_files]

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            df = pd.read_csv(p)
            for _, r in df.iterrows():
                home = str(r.get("HomeTeam") or "")
                away = str(r.get("AwayTeam") or "")
                ftr = str(r.get("FTR") or "").upper().strip()  # H/D/A
                if not home or not away or ftr not in {"H", "D", "A"}:
                    continue

                # open/close proxies per outcome
                oh_open = _first_num(r, ["B365H", "WHH", "VCH", "IWH", "LBH"])
                od_open = _first_num(r, ["B365D", "WHD", "VCD", "IWD", "LBD"])
                oa_open = _first_num(r, ["B365A", "WHA", "VCA", "IWA", "LBA"])

                oh_close = _first_num(r, ["PSH", "AvgH", "MaxH", "B365H"])
                od_close = _first_num(r, ["PSD", "AvgD", "MaxD", "B365D"])
                oa_close = _first_num(r, ["PSA", "AvgA", "MaxA", "B365A"])

                cand = [("H", oh_close), ("D", od_close), ("A", oa_close)]
                cand = [(k, v) for k, v in cand if v is not None]
                if not cand:
                    continue

                pick, odds_close = sorted(cand, key=lambda x: x[1])[0]
                selection = home if pick == "H" else away if pick == "A" else "Draw"
                status = "won" if pick == ftr else "lost"

                odds_open_map = {"H": oh_open, "D": od_open, "A": oa_open}
                odds_open = odds_open_map.get(pick) or odds_close

                stake = 1.0
                pnl = round((odds_close - 1.0) if status == "won" else -stake, 2)

                event_id = _mk_event_id(["football", r.get("Div"), r.get("Date"), home, away])
                key = (event_id, selection)
                if key in existing:
                    continue

                db.add(
                    PlacedBet(
                        event_id=event_id,
                        sport=f"soccer_{str(r.get('Div') or '').lower()}",
                        market="h2h",
                        selection=selection,
                        odds=float(odds_close),
                        odds_open=float(odds_open),
                        odds_close=float(odds_close),
                        clv=float(odds_open) - float(odds_close),
                        stake=stake,
                        status=status,
                        pnl=pnl,
                    )
                )
                existing.add(key)
                inserted += 1

        db.commit()
    return inserted


def import_tennis(folder: Path, limit_files: int = 0) -> int:
    files = sorted(folder.glob("*.xlsx"))
    if limit_files > 0:
        files = files[:limit_files]

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            df = pd.read_excel(p)
            for _, r in df.iterrows():
                winner = str(r.get("Winner") or "")
                loser = str(r.get("Loser") or "")
                if not winner or not loser:
                    continue

                w_open = _first_num(r, ["B365W", "CBW", "EXW"])
                l_open = _first_num(r, ["B365L", "CBL", "EXL"])
                w_close = _first_num(r, ["PSW", "AvgW", "MaxW", "B365W"])
                l_close = _first_num(r, ["PSL", "AvgL", "MaxL", "B365L"])
                if w_close is None and l_close is None:
                    continue

                # pick favorite between winner/loser by close odds
                if w_close is not None and (l_close is None or w_close <= l_close):
                    selection, odds_close = winner, w_close
                    odds_open = w_open or w_close
                    status = "won"
                else:
                    selection, odds_close = loser, l_close
                    odds_open = l_open or l_close
                    status = "lost"

                stake = 1.0
                pnl = round((odds_close - 1.0) if status == "won" else -stake, 2)
                event_id = _mk_event_id([
                    "tennis",
                    p.name,
                    r.get("Date"),
                    r.get("Tournament"),
                    winner,
                    loser,
                ])
                sport_prefix = "wta" if "-2" in p.stem else "atp"
                key = (event_id, selection)
                if key in existing:
                    continue

                db.add(
                    PlacedBet(
                        event_id=event_id,
                        sport=f"tennis_{sport_prefix}",
                        market="h2h",
                        selection=selection,
                        odds=float(odds_close),
                        odds_open=float(odds_open),
                        odds_close=float(odds_close),
                        clv=float(odds_open) - float(odds_close),
                        stake=stake,
                        status=status,
                        pnl=pnl,
                    )
                )
                existing.add(key)
                inserted += 1

        db.commit()
    return inserted


def import_nba(folder: Path, max_rows: int = 50000) -> int:
    p = folder / "Games.csv"
    if not p.exists():
        return 0

    df = pd.read_csv(p)
    if max_rows > 0 and len(df) > max_rows:
        df = df.tail(max_rows)

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for _, r in df.iterrows():
            home = f"{str(r.get('hometeamCity') or '').strip()} {str(r.get('hometeamName') or '').strip()}".strip()
            away = f"{str(r.get('awayteamCity') or '').strip()} {str(r.get('awayteamName') or '').strip()}".strip()
            try:
                hs = int(r.get("homeScore"))
                a_s = int(r.get("awayScore"))
            except Exception:
                continue
            if not home or not away:
                continue

            selection = home  # fixed historical policy for balanced label creation
            status = "won" if hs > a_s else "lost"
            odds = 1.91
            stake = 1.0
            pnl = round((odds - 1.0) if status == "won" else -stake, 2)

            event_id = _mk_event_id(["nba", r.get("gameId") or "", r.get("gameDateTimeEst") or "", home, away])
            key = (event_id, selection)
            if key in existing:
                continue

            db.add(
                PlacedBet(
                    event_id=event_id,
                    sport="basketball_nba",
                    market="h2h",
                    selection=selection,
                    odds=odds,
                    odds_open=odds,
                    odds_close=odds,
                    clv=0.0,
                    stake=stake,
                    status=status,
                    pnl=pnl,
                )
            )
            existing.add(key)
            inserted += 1

        db.commit()

    return inserted


def main():
    ap = argparse.ArgumentParser(description="Import historical football/tennis/nba results into placed_bets")
    ap.add_argument("--imports-dir", default="data/imports")
    ap.add_argument("--football-files", type=int, default=0, help="limit football CSV files (0=all)")
    ap.add_argument("--tennis-files", type=int, default=0, help="limit tennis XLSX files (0=all)")
    ap.add_argument("--nba-max-rows", type=int, default=50000)
    args = ap.parse_args()

    Base.metadata.create_all(bind=engine)

    base = Path(args.imports_dir)
    f = import_football(base / "football", limit_files=args.football_files)
    t = import_tennis(base / "tennis", limit_files=args.tennis_files)
    n = import_nba(base / "nba", max_rows=args.nba_max_rows)

    with SessionLocal() as db:
        total = db.scalar(select(func.count()).select_from(PlacedBet))

    print({"football": f, "tennis": t, "nba": n, "total_after": total})


if __name__ == "__main__":
    main()
