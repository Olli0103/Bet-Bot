#!/usr/bin/env python3
"""Intelligent Universal Data Importer for historical sports results.

Supports:
- Football (soccer): H2H, Totals (Over 1.5/2.5), BTTS from football-data.co.uk CSVs
- Tennis: ATP, WTA, Challenger from .csv and .xlsx files
- US Sports (NBA, NFL, NHL): Moneyline, Spreads, Totals from aussportsbetting CSVs

All sync DB calls are safe because this runs as a standalone script.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select, func

from src.data.models import Base, PlacedBet
from src.data.postgres import SessionLocal, engine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_event_id(parts: list[str]) -> str:
    s = "|".join([str(x or "") for x in parts])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _first_num(row, cols: list[str]) -> Optional[float]:
    for c in cols:
        if c in row and pd.notna(row[c]):
            try:
                v = float(row[c])
                if v > 1.0:
                    return v
            except Exception:
                pass
    return None


def _first_str(row, cols: list[str]) -> Optional[str]:
    """Return first non-empty string value from candidate column names."""
    for c in cols:
        if c in row and pd.notna(row[c]):
            val = str(row[c]).strip()
            if val:
                return val
    return None


def _first_int(row, cols: list[str]) -> Optional[int]:
    """Return first valid integer from candidate column names."""
    for c in cols:
        if c in row and pd.notna(row[c]):
            try:
                return int(float(row[c]))
            except (ValueError, TypeError):
                pass
    return None


def _first_float(row, cols: list[str]) -> Optional[float]:
    """Return first valid float from candidate column names."""
    for c in cols:
        if c in row and pd.notna(row[c]):
            try:
                return float(row[c])
            except (ValueError, TypeError):
                pass
    return None


# ---------------------------------------------------------------------------
# Football (Soccer): H2H + Totals + BTTS
# ---------------------------------------------------------------------------

def import_football(folder: Path, limit_files: int = 0) -> int:
    """Import football CSVs generating H2H, Totals, and BTTS training rows."""
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
                ftr = str(r.get("FTR") or "").upper().strip()
                if not home or not away or ftr not in {"H", "D", "A"}:
                    continue

                div_str = str(r.get("Div") or "").lower()
                sport = f"soccer_{div_str}"
                base_event_id = _mk_event_id(["football", r.get("Div"), r.get("Date"), home, away])

                # --- H2H market ---
                oh_open = _first_num(r, ["B365H", "WHH", "VCH", "IWH", "LBH"])
                od_open = _first_num(r, ["B365D", "WHD", "VCD", "IWD", "LBD"])
                oa_open = _first_num(r, ["B365A", "WHA", "VCA", "IWA", "LBA"])

                oh_close = _first_num(r, ["PSH", "AvgH", "MaxH", "B365H"])
                od_close = _first_num(r, ["PSD", "AvgD", "MaxD", "B365D"])
                oa_close = _first_num(r, ["PSA", "AvgA", "MaxA", "B365A"])

                cand = [("H", oh_close), ("D", od_close), ("A", oa_close)]
                cand = [(k, v) for k, v in cand if v is not None]
                if cand:
                    pick, odds_close = sorted(cand, key=lambda x: x[1])[0]
                    selection = home if pick == "H" else away if pick == "A" else "Draw"
                    status = "won" if pick == ftr else "lost"
                    odds_open_map = {"H": oh_open, "D": od_open, "A": oa_open}
                    odds_open = odds_open_map.get(pick) or odds_close
                    stake = 1.0
                    pnl = round((odds_close - 1.0) if status == "won" else -stake, 2)

                    key = (base_event_id, selection)
                    if key not in existing:
                        db.add(PlacedBet(
                            event_id=base_event_id, sport=sport, market="h2h",
                            selection=selection, odds=float(odds_close),
                            odds_open=float(odds_open), odds_close=float(odds_close),
                            clv=(float(odds_open) / float(odds_close)) - 1.0 if float(odds_close) > 1.0 else 0.0,
                            stake=stake, status=status, pnl=pnl,
                        ))
                        existing.add(key)
                        inserted += 1

                # --- Totals + BTTS (requires actual goals) ---
                if "FTHG" in r.index and "FTAG" in r.index and pd.notna(r.get("FTHG")) and pd.notna(r.get("FTAG")):
                    fthg = int(float(r["FTHG"]))
                    ftag = int(float(r["FTAG"]))
                    total_goals = fthg + ftag

                    # Over 1.5 Goals
                    over_15_odds = _first_num(r, ["BbAv>2.5", "Avg>2.5"]) or 1.40
                    sel_15 = "Over 1.5"
                    status_15 = "won" if total_goals > 1 else "lost"
                    eid_15 = _mk_event_id(["football", r.get("Div"), r.get("Date"), home, away, "o15"])
                    key_15 = (eid_15, sel_15)
                    if key_15 not in existing:
                        pnl_15 = round((over_15_odds - 1.0) if status_15 == "won" else -1.0, 2)
                        db.add(PlacedBet(
                            event_id=eid_15, sport=sport, market="totals",
                            selection=sel_15, odds=over_15_odds,
                            stake=1.0, status=status_15, pnl=pnl_15,
                        ))
                        existing.add(key_15)
                        inserted += 1

                    # Over 2.5 Goals
                    over_25_odds = _first_num(r, ["BbAv>2.5", "Avg>2.5", "Max>2.5", "B365>2.5", "P>2.5"]) or 1.85
                    sel_25 = "Over 2.5"
                    status_25 = "won" if total_goals > 2 else "lost"
                    eid_25 = _mk_event_id(["football", r.get("Div"), r.get("Date"), home, away, "o25"])
                    key_25 = (eid_25, sel_25)
                    if key_25 not in existing:
                        pnl_25 = round((over_25_odds - 1.0) if status_25 == "won" else -1.0, 2)
                        db.add(PlacedBet(
                            event_id=eid_25, sport=sport, market="totals",
                            selection=sel_25, odds=over_25_odds,
                            stake=1.0, status=status_25, pnl=pnl_25,
                        ))
                        existing.add(key_25)
                        inserted += 1

                    # BTTS (Both Teams To Score)
                    btts_odds = _first_num(r, ["BbAvBTTS", "AvgBTTS", "MaxBTTS"]) or 1.75
                    sel_btts = "BTTS Yes"
                    status_btts = "won" if fthg >= 1 and ftag >= 1 else "lost"
                    eid_btts = _mk_event_id(["football", r.get("Div"), r.get("Date"), home, away, "btts"])
                    key_btts = (eid_btts, sel_btts)
                    if key_btts not in existing:
                        pnl_btts = round((btts_odds - 1.0) if status_btts == "won" else -1.0, 2)
                        db.add(PlacedBet(
                            event_id=eid_btts, sport=sport, market="btts",
                            selection=sel_btts, odds=btts_odds,
                            stake=1.0, status=status_btts, pnl=pnl_btts,
                        ))
                        existing.add(key_btts)
                        inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# Tennis: ATP, WTA, Challenger (.csv + .xlsx)
# ---------------------------------------------------------------------------

def import_tennis(folder: Path, limit_files: int = 0) -> int:
    """Import tennis files handling ATP, WTA, and Challenger formats."""
    csv_files = sorted(folder.glob("*.csv"))
    xlsx_files = sorted(folder.glob("*.xlsx"))
    xls_files = sorted(folder.glob("*.xls"))
    files = csv_files + xlsx_files + xls_files
    if limit_files > 0:
        files = files[:limit_files]

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            try:
                if p.suffix == ".csv":
                    df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
                else:
                    df = pd.read_excel(p)
            except Exception:
                continue

            # Detect sport prefix from filename
            stem_lower = p.stem.lower()
            if "wta" in stem_lower or "-2" in stem_lower or "women" in stem_lower:
                sport_prefix = "wta"
            elif "challenger" in stem_lower or "chall" in stem_lower or "ch_" in stem_lower:
                sport_prefix = "challenger"
            else:
                sport_prefix = "atp"

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
                    "tennis", p.name, r.get("Date"),
                    r.get("Tournament"), winner, loser,
                ])
                key = (event_id, selection)
                if key in existing:
                    continue

                db.add(PlacedBet(
                    event_id=event_id,
                    sport=f"tennis_{sport_prefix}",
                    market="h2h",
                    selection=selection,
                    odds=float(odds_close),
                    odds_open=float(odds_open),
                    odds_close=float(odds_close),
                    clv=(float(odds_open) / float(odds_close)) - 1.0 if float(odds_close) > 1.0 else 0.0,
                    stake=stake, status=status, pnl=pnl,
                ))
                existing.add(key)
                inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# US Sports: NBA, NFL, NHL — Aussportsbetting CSV format
# ---------------------------------------------------------------------------

def import_us_sports(folder: Path, sport_label: str, max_rows: int = 50000) -> int:
    """Universal importer for aussportsbetting.com CSV format.

    Generates separate PlacedBet rows for Moneyline, Spreads, and Totals.
    """
    files = sorted(folder.glob("*.csv"))
    if not files:
        return 0

    HOME_COLS = ["Home Team", "HomeTeam", "Home", "home_team"]
    AWAY_COLS = ["Away Team", "AwayTeam", "Away", "away_team"]
    HOME_SCORE_COLS = ["Home Score", "HomeScore", "homeScore", "Home Points", "PtsH"]
    AWAY_SCORE_COLS = ["Away Score", "AwayScore", "awayScore", "Away Points", "PtsA"]
    HOME_ML_COLS = ["Home Odds", "HomeOdds", "Home Line Close", "PinnH", "B365H"]
    AWAY_ML_COLS = ["Away Odds", "AwayOdds", "Away Line Close", "PinnA", "B365A"]
    SPREAD_LINE_COLS = ["Home Line", "HomeLine", "HomeSpread", "Spread"]
    SPREAD_HOME_ODDS_COLS = ["Home Line Odds", "HomeLineOdds", "Home Spread Odds"]
    SPREAD_AWAY_ODDS_COLS = ["Away Line Odds", "AwayLineOdds", "Away Spread Odds"]
    TOTAL_LINE_COLS = ["Total Score Line", "TotalLine", "OU", "Total", "OverUnder"]
    OVER_ODDS_COLS = ["Total Score Over Odds", "OverOdds", "Over Odds"]
    UNDER_ODDS_COLS = ["Total Score Under Odds", "UnderOdds", "Under Odds"]
    DATE_COLS = ["Date", "date", "GameDate", "gameDateTimeEst"]
    GAME_ID_COLS = ["Game ID", "gameId", "game_id", "ID"]

    sport_key_map = {
        "nba": "basketball_nba",
        "nfl": "americanfootball_nfl",
        "nhl": "icehockey_nhl",
    }
    sport_key = sport_key_map.get(sport_label, f"sport_{sport_label}")

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            try:
                df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
            except Exception:
                continue
            if max_rows > 0 and len(df) > max_rows:
                df = df.tail(max_rows)

            for _, r in df.iterrows():
                home = str(_first_str(r, HOME_COLS) or "").strip()
                away = str(_first_str(r, AWAY_COLS) or "").strip()
                if not home or not away:
                    continue

                hs = _first_int(r, HOME_SCORE_COLS)
                a_s = _first_int(r, AWAY_SCORE_COLS)
                if hs is None or a_s is None:
                    continue

                date_str = _first_str(r, DATE_COLS) or ""
                game_id = _first_str(r, GAME_ID_COLS) or ""

                # --- MONEYLINE (H2H) ---
                home_ml_odds = _first_num(r, HOME_ML_COLS)
                away_ml_odds = _first_num(r, AWAY_ML_COLS)

                if home_ml_odds is not None:
                    ml_eid = _mk_event_id([sport_label, game_id, date_str, home, away, "ml"])
                    ml_sel = home
                    ml_status = "won" if hs > a_s else "lost"
                    ml_pnl = round((home_ml_odds - 1.0) if ml_status == "won" else -1.0, 2)
                    key = (ml_eid, ml_sel)
                    if key not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid, sport=sport_key, market="h2h",
                            selection=ml_sel, odds=float(home_ml_odds),
                            stake=1.0, status=ml_status, pnl=ml_pnl,
                        ))
                        existing.add(key)
                        inserted += 1

                if away_ml_odds is not None:
                    ml_eid_a = _mk_event_id([sport_label, game_id, date_str, home, away, "ml_a"])
                    ml_sel_a = away
                    ml_status_a = "won" if a_s > hs else "lost"
                    ml_pnl_a = round((away_ml_odds - 1.0) if ml_status_a == "won" else -1.0, 2)
                    key_a = (ml_eid_a, ml_sel_a)
                    if key_a not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid_a, sport=sport_key, market="h2h",
                            selection=ml_sel_a, odds=float(away_ml_odds),
                            stake=1.0, status=ml_status_a, pnl=ml_pnl_a,
                        ))
                        existing.add(key_a)
                        inserted += 1

                # --- SPREADS (Handicap) ---
                spread_line = _first_float(r, SPREAD_LINE_COLS)
                spread_h_odds = _first_num(r, SPREAD_HOME_ODDS_COLS)

                if spread_line is not None and spread_h_odds is not None:
                    sp_eid = _mk_event_id([sport_label, game_id, date_str, home, away, "sp"])
                    home_adjusted = hs + spread_line
                    sp_status = "won" if home_adjusted > a_s else ("lost" if home_adjusted < a_s else "void")
                    sp_sel = f"{home} {spread_line:+.1f}"
                    sp_pnl = round(
                        (spread_h_odds - 1.0) if sp_status == "won"
                        else (-1.0 if sp_status == "lost" else 0.0), 2
                    )
                    key_sp = (sp_eid, sp_sel)
                    if key_sp not in existing:
                        db.add(PlacedBet(
                            event_id=sp_eid, sport=sport_key, market="spreads",
                            selection=sp_sel, odds=float(spread_h_odds),
                            stake=1.0, status=sp_status, pnl=sp_pnl,
                        ))
                        existing.add(key_sp)
                        inserted += 1

                # --- TOTALS (Over/Under) ---
                total_line = _first_float(r, TOTAL_LINE_COLS)
                over_odds = _first_num(r, OVER_ODDS_COLS)
                under_odds = _first_num(r, UNDER_ODDS_COLS)

                if total_line is not None and over_odds is not None:
                    actual_total = hs + a_s

                    # Over
                    tot_o_eid = _mk_event_id([sport_label, game_id, date_str, home, away, f"o{total_line}"])
                    tot_o_sel = f"Over {total_line}"
                    tot_o_status = "won" if actual_total > total_line else ("lost" if actual_total < total_line else "void")
                    tot_o_pnl = round(
                        (over_odds - 1.0) if tot_o_status == "won"
                        else (-1.0 if tot_o_status == "lost" else 0.0), 2
                    )
                    key_to = (tot_o_eid, tot_o_sel)
                    if key_to not in existing:
                        db.add(PlacedBet(
                            event_id=tot_o_eid, sport=sport_key, market="totals",
                            selection=tot_o_sel, odds=float(over_odds),
                            stake=1.0, status=tot_o_status, pnl=tot_o_pnl,
                        ))
                        existing.add(key_to)
                        inserted += 1

                    # Under
                    if under_odds is not None:
                        tot_u_eid = _mk_event_id([sport_label, game_id, date_str, home, away, f"u{total_line}"])
                        tot_u_sel = f"Under {total_line}"
                        tot_u_status = "won" if actual_total < total_line else ("lost" if actual_total > total_line else "void")
                        tot_u_pnl = round(
                            (under_odds - 1.0) if tot_u_status == "won"
                            else (-1.0 if tot_u_status == "lost" else 0.0), 2
                        )
                        key_tu = (tot_u_eid, tot_u_sel)
                        if key_tu not in existing:
                            db.add(PlacedBet(
                                event_id=tot_u_eid, sport=sport_key, market="totals",
                                selection=tot_u_sel, odds=float(under_odds),
                                stake=1.0, status=tot_u_status, pnl=tot_u_pnl,
                            ))
                            existing.add(key_tu)
                            inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Import historical results (football, tennis, NBA, NFL, NHL) into placed_bets"
    )
    ap.add_argument("--imports-dir", default="data/imports")
    ap.add_argument("--football-files", type=int, default=0, help="limit football CSV files (0=all)")
    ap.add_argument("--tennis-files", type=int, default=0, help="limit tennis files (0=all)")
    ap.add_argument("--nba-dir", type=str, default=None, help="path to NBA CSV directory")
    ap.add_argument("--nfl-dir", type=str, default=None, help="path to NFL CSV directory")
    ap.add_argument("--nhl-dir", type=str, default=None, help="path to NHL CSV directory")
    ap.add_argument("--max-rows", type=int, default=50000, help="max rows per US sport (0=all)")
    args = ap.parse_args()

    Base.metadata.create_all(bind=engine)

    base = Path(args.imports_dir)
    counts = {}

    fb_dir = base / "football"
    counts["football"] = import_football(fb_dir, limit_files=args.football_files) if fb_dir.exists() else 0

    tn_dir = base / "tennis"
    counts["tennis"] = import_tennis(tn_dir, limit_files=args.tennis_files) if tn_dir.exists() else 0

    nba_dir = Path(args.nba_dir) if args.nba_dir else base / "nba"
    counts["nba"] = import_us_sports(nba_dir, "nba", max_rows=args.max_rows) if nba_dir.exists() else 0

    nfl_dir = Path(args.nfl_dir) if args.nfl_dir else base / "nfl"
    counts["nfl"] = import_us_sports(nfl_dir, "nfl", max_rows=args.max_rows) if nfl_dir.exists() else 0

    nhl_dir = Path(args.nhl_dir) if args.nhl_dir else base / "nhl"
    counts["nhl"] = import_us_sports(nhl_dir, "nhl", max_rows=args.max_rows) if nhl_dir.exists() else 0

    with SessionLocal() as db:
        total = db.scalar(select(func.count()).select_from(PlacedBet))
    counts["total_after"] = total
    print(counts)


if __name__ == "__main__":
    main()
