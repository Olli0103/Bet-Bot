#!/usr/bin/env python3
"""Intelligent Universal Data Importer for historical sports results.

Supports:
- Football (soccer): H2H, Totals (Over 1.5/2.5), BTTS from football-data.co.uk CSVs
  Maps ``Div`` column (D1, E0, SP1 …) via SPORT_MAPPING to Odds API keys.
- Tennis: ATP, WTA, Challenger from .csv and .xlsx files
- US Sports:
  - NBA (basketball): Kaggle-style one-row-per-game, American odds
  - NFL (american football): aussportsbetting format, decimal odds
  - NHL (ice hockey): one-row-per-team-per-game, American odds

All sync DB calls are safe because this runs as a standalone script.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select, func

from src.core.sport_mapping import csv_code_to_api_key
from src.data.models import Base, PlacedBet
from src.data.postgres import SessionLocal, engine

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mk_event_id(parts: list[str]) -> str:
    """Deterministic event ID hash from component strings."""
    s = "|".join([str(x or "") for x in parts])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _first_num(row, cols: list[str]) -> Optional[float]:
    """Return first numeric value > 1.0 from candidate column names (odds)."""
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            try:
                v = float(row[c])
                if v > 1.0:
                    return v
            except (ValueError, TypeError):
                pass
    return None


def _first_str(row, cols: list[str]) -> Optional[str]:
    """Return first non-empty string value from candidate column names."""
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            val = str(row[c]).strip()
            if val:
                return val
    return None


def _first_int(row, cols: list[str]) -> Optional[int]:
    """Return first valid integer from candidate column names."""
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            try:
                return int(float(row[c]))
            except (ValueError, TypeError):
                pass
    return None


def _first_float(row, cols: list[str]) -> Optional[float]:
    """Return first valid float from candidate column names."""
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except (ValueError, TypeError):
                pass
    return None


def _american_to_decimal(american: float) -> Optional[float]:
    """Convert American moneyline odds to decimal.

    +150 -> 2.50, -200 -> 1.50.  Returns None on invalid input.
    """
    if american == 0:
        return None
    if american > 0:
        return round(1.0 + american / 100.0, 4)
    else:
        return round(1.0 + 100.0 / abs(american), 4)


def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    """Read a CSV with lenient error handling."""
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", **kwargs)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip", **kwargs)
        except Exception as exc:
            log.warning("Cannot read %s: %s", path, exc)
            return None


# ---------------------------------------------------------------------------
# Football (Soccer): H2H + Totals + BTTS
# Maps Div column -> SPORT_MAPPING -> Odds API key
# ---------------------------------------------------------------------------

def import_football(folder: Path, limit_files: int = 0) -> int:
    """Import football CSVs generating H2H, Totals, and BTTS training rows.

    Uses ``Div`` column (e.g. D1, E0, SP1) mapped through SPORT_MAPPING
    to produce the correct Odds API sport key for each row.
    """
    files = sorted(folder.glob("*.csv"))
    if limit_files > 0:
        files = files[:limit_files]

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue

            for _, r in df.iterrows():
                home = str(r.get("HomeTeam") or r.get("HT") or "").strip()
                away = str(r.get("AwayTeam") or r.get("AT") or "").strip()
                ftr = str(r.get("FTR") or "").upper().strip()
                if not home or not away or ftr not in {"H", "D", "A"}:
                    continue

                # --- Resolve sport key via central mapping ---
                div_raw = str(r.get("Div") or "").strip()
                sport = csv_code_to_api_key(div_raw)
                if sport is None:
                    # Fallback: construct a best-effort key
                    sport = f"soccer_{div_raw.lower()}" if div_raw else "soccer_unknown"

                base_event_id = _mk_event_id(["football", div_raw, r.get("Date"), home, away])

                # --- H2H market ---
                # Opening odds (B365 or similar)
                oh_open = _first_num(r, ["B365H", "WHH", "VCH", "IWH", "LBH"])
                od_open = _first_num(r, ["B365D", "WHD", "VCD", "IWD", "LBD"])
                oa_open = _first_num(r, ["B365A", "WHA", "VCA", "IWA", "LBA"])

                # Closing odds (Pinnacle/Avg as proxy)
                oh_close = _first_num(r, ["PSH", "PSCH", "AvgH", "MaxH", "B365H"])
                od_close = _first_num(r, ["PSD", "PSCD", "AvgD", "MaxD", "B365D"])
                oa_close = _first_num(r, ["PSA", "PSCA", "AvgA", "MaxA", "B365A"])

                cand = [("H", oh_close), ("D", od_close), ("A", oa_close)]
                cand = [(k, v) for k, v in cand if v is not None]
                if cand:
                    # Pick the favourite (lowest closing odds)
                    pick, odds_close = sorted(cand, key=lambda x: x[1])[0]
                    selection = home if pick == "H" else away if pick == "A" else "Draw"
                    status = "won" if pick == ftr else "lost"
                    odds_open_map = {"H": oh_open, "D": od_open, "A": oa_open}
                    odds_open = odds_open_map.get(pick) or odds_close
                    pnl = round((odds_close - 1.0) if status == "won" else -1.0, 2)

                    key = (base_event_id, selection)
                    if key not in existing:
                        db.add(PlacedBet(
                            event_id=base_event_id, sport=sport, market="h2h",
                            selection=selection, odds=float(odds_close),
                            odds_open=float(odds_open), odds_close=float(odds_close),
                            clv=round((float(odds_open) / float(odds_close)) - 1.0, 4)
                                if float(odds_close) > 1.0 else 0.0,
                            stake=1.0, status=status, pnl=pnl,
                        ))
                        existing.add(key)
                        inserted += 1

                # --- Totals + BTTS (requires actual goals) ---
                fthg_raw = r.get("FTHG")
                ftag_raw = r.get("FTAG")
                if pd.notna(fthg_raw) and pd.notna(ftag_raw):
                    fthg = int(float(fthg_raw))
                    ftag = int(float(ftag_raw))
                    total_goals = fthg + ftag

                    # Over 2.5 Goals (explicit B365 column preferred)
                    over_25_odds = _first_num(r, [
                        "B365>2.5", "P>2.5", "BbAv>2.5", "Avg>2.5", "Max>2.5",
                    ]) or 1.85
                    under_25_odds = _first_num(r, [
                        "B365<2.5", "P<2.5", "BbAv<2.5", "Avg<2.5", "Max<2.5",
                    ]) or 1.95

                    # -- Over 2.5
                    sel_25 = "Over 2.5"
                    status_25 = "won" if total_goals > 2 else "lost"
                    eid_25 = _mk_event_id(["football", div_raw, r.get("Date"), home, away, "o25"])
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

                    # -- Under 2.5
                    sel_u25 = "Under 2.5"
                    status_u25 = "won" if total_goals < 3 else "lost"
                    eid_u25 = _mk_event_id(["football", div_raw, r.get("Date"), home, away, "u25"])
                    key_u25 = (eid_u25, sel_u25)
                    if key_u25 not in existing:
                        pnl_u25 = round((under_25_odds - 1.0) if status_u25 == "won" else -1.0, 2)
                        db.add(PlacedBet(
                            event_id=eid_u25, sport=sport, market="totals",
                            selection=sel_u25, odds=under_25_odds,
                            stake=1.0, status=status_u25, pnl=pnl_u25,
                        ))
                        existing.add(key_u25)
                        inserted += 1

                    # -- Over 1.5 (derive from Over 2.5 if no explicit column)
                    # Mathematical approximation: O1.5 is ~30% more likely than O2.5
                    over_15_odds = _first_num(r, [
                        "B365>1.5", "Avg>1.5", "Max>1.5",
                    ])
                    if over_15_odds is None and over_25_odds > 1.0:
                        ip_25 = 1.0 / over_25_odds
                        ip_15 = min(0.95, ip_25 + 0.20)
                        over_15_odds = round(1.0 / ip_15, 2) if ip_15 > 0.01 else 1.10

                    if over_15_odds and over_15_odds > 1.0:
                        sel_15 = "Over 1.5"
                        status_15 = "won" if total_goals > 1 else "lost"
                        eid_15 = _mk_event_id(["football", div_raw, r.get("Date"), home, away, "o15"])
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

                    # BTTS (Both Teams To Score)
                    btts_odds = _first_num(r, [
                        "BbAvBTTS", "AvgBTTS", "MaxBTTS",
                    ]) or 1.75
                    sel_btts = "BTTS Yes"
                    status_btts = "won" if fthg >= 1 and ftag >= 1 else "lost"
                    eid_btts = _mk_event_id(["football", div_raw, r.get("Date"), home, away, "btts"])
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
    """Import tennis files handling ATP, WTA, and Challenger formats.

    Detects tour from filename patterns and maps via SPORT_MAPPING.
    Generates Winner *and* Loser rows so the ML model sees both sides.
    """
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

            # Detect tour from filename
            stem_lower = p.stem.lower()
            if "wta" in stem_lower or "women" in stem_lower:
                tour_code = "wta"
            elif "challenger" in stem_lower or "chall" in stem_lower or "ch_" in stem_lower:
                tour_code = "challenger"
            else:
                tour_code = "atp"

            sport = csv_code_to_api_key(tour_code) or f"tennis_{tour_code}"

            for _, r in df.iterrows():
                winner = str(r.get("Winner") or "").strip()
                loser = str(r.get("Loser") or "").strip()
                if not winner or not loser:
                    continue

                # Odds columns
                w_open = _first_num(r, ["B365W", "CBW", "EXW", "PSW"])
                l_open = _first_num(r, ["B365L", "CBL", "EXL", "PSL"])
                w_close = _first_num(r, ["PSW", "AvgW", "MaxW", "B365W"])
                l_close = _first_num(r, ["PSL", "AvgL", "MaxL", "B365L"])
                if w_close is None and l_close is None:
                    continue

                event_id = _mk_event_id([
                    "tennis", p.name, r.get("Date"),
                    r.get("Tournament"), winner, loser,
                ])

                # Determine who was the pre-match favourite
                if w_close is not None and l_close is not None:
                    winner_is_fav = w_close <= l_close
                elif w_close is not None:
                    winner_is_fav = True
                else:
                    winner_is_fav = False

                # --- Winner row (always status=won) ---
                if w_close is not None:
                    w_odds_open = w_open or w_close
                    w_pnl = round(w_close - 1.0, 2)
                    w_clv = round((w_odds_open / w_close) - 1.0, 4) if w_close > 1.0 else 0.0
                    key_w = (event_id, winner)
                    if key_w not in existing:
                        db.add(PlacedBet(
                            event_id=event_id, sport=sport, market="h2h",
                            selection=winner, odds=float(w_close),
                            odds_open=float(w_odds_open), odds_close=float(w_close),
                            clv=w_clv, stake=1.0, status="won", pnl=w_pnl,
                        ))
                        existing.add(key_w)
                        inserted += 1

                # --- Loser row (always status=lost) ---
                if l_close is not None:
                    l_odds_open = l_open or l_close
                    l_clv = round((l_odds_open / l_close) - 1.0, 4) if l_close > 1.0 else 0.0
                    key_l = (event_id, loser)
                    if key_l not in existing:
                        db.add(PlacedBet(
                            event_id=event_id, sport=sport, market="h2h",
                            selection=loser, odds=float(l_close),
                            odds_open=float(l_odds_open), odds_close=float(l_close),
                            clv=l_clv, stake=1.0, status="lost", pnl=-1.0,
                        ))
                        existing.add(key_l)
                        inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# NBA — Kaggle-style one-row-per-game, American moneyline odds
# ---------------------------------------------------------------------------

def import_nba(folder: Path, max_rows: int = 50000) -> int:
    """Import NBA game data.

    Expected columns (Kaggle NBA odds format):
      game_date, team_home, team_away, pts_home, pts_away,
      moneyline_home, moneyline_away, spread, total
    Odds are in American format (-110, +150, etc.).
    """
    sport = csv_code_to_api_key("nba") or "basketball_nba"
    files = sorted(folder.glob("*.csv"))
    if not files:
        return 0

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue
            if max_rows > 0 and len(df) > max_rows:
                df = df.tail(max_rows)

            for _, r in df.iterrows():
                home = _first_str(r, [
                    "team_home", "Home Team", "HomeTeam", "Home", "home_team",
                ]) or ""
                away = _first_str(r, [
                    "team_away", "Away Team", "AwayTeam", "Away", "away_team",
                ]) or ""
                if not home or not away:
                    continue

                pts_h = _first_int(r, ["pts_home", "Home Score", "HomeScore", "PtsH", "Home Points"])
                pts_a = _first_int(r, ["pts_away", "Away Score", "AwayScore", "PtsA", "Away Points"])
                if pts_h is None or pts_a is None:
                    continue

                date_str = _first_str(r, ["game_date", "Date", "date", "GameDate"]) or ""
                game_id = _first_str(r, ["game_id", "Game ID", "gameId", "ID"]) or ""

                # --- MONEYLINE (H2H) — American odds ---
                ml_home_raw = _first_float(r, ["moneyline_home", "Home ML", "HomeML", "Home Odds"])
                ml_away_raw = _first_float(r, ["moneyline_away", "Away ML", "AwayML", "Away Odds"])

                home_ml_dec = _american_to_decimal(ml_home_raw) if ml_home_raw is not None else None
                away_ml_dec = _american_to_decimal(ml_away_raw) if ml_away_raw is not None else None

                # Also try decimal columns directly (aussportsbetting fallback)
                if home_ml_dec is None:
                    home_ml_dec = _first_num(r, ["Home Odds Close", "Home Odds", "PinnH", "B365H"])
                if away_ml_dec is None:
                    away_ml_dec = _first_num(r, ["Away Odds Close", "Away Odds", "PinnA", "B365A"])

                if home_ml_dec is not None and home_ml_dec > 1.0:
                    ml_eid = _mk_event_id(["nba", game_id, date_str, home, away, "ml_h"])
                    ml_status = "won" if pts_h > pts_a else "lost"
                    key = (ml_eid, home)
                    if key not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid, sport=sport, market="h2h",
                            selection=home, odds=float(home_ml_dec),
                            stake=1.0, status=ml_status,
                            pnl=round((home_ml_dec - 1.0) if ml_status == "won" else -1.0, 2),
                        ))
                        existing.add(key)
                        inserted += 1

                if away_ml_dec is not None and away_ml_dec > 1.0:
                    ml_eid_a = _mk_event_id(["nba", game_id, date_str, home, away, "ml_a"])
                    ml_status_a = "won" if pts_a > pts_h else "lost"
                    key_a = (ml_eid_a, away)
                    if key_a not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid_a, sport=sport, market="h2h",
                            selection=away, odds=float(away_ml_dec),
                            stake=1.0, status=ml_status_a,
                            pnl=round((away_ml_dec - 1.0) if ml_status_a == "won" else -1.0, 2),
                        ))
                        existing.add(key_a)
                        inserted += 1

                # --- SPREADS ---
                spread_raw = _first_float(r, [
                    "spread", "Home Line Close", "HomeSpread", "Home Line",
                ])
                spread_odds_dec = 1.91  # NBA standard vig for both sides
                if spread_raw is not None:
                    sp_eid = _mk_event_id(["nba", game_id, date_str, home, away, "sp"])
                    home_adjusted = pts_h + spread_raw
                    sp_status = (
                        "won" if home_adjusted > pts_a
                        else "lost" if home_adjusted < pts_a
                        else "void"
                    )
                    sp_sel = f"{home} {spread_raw:+.1f}"
                    key_sp = (sp_eid, sp_sel)
                    if key_sp not in existing:
                        db.add(PlacedBet(
                            event_id=sp_eid, sport=sport, market="spreads",
                            selection=sp_sel, odds=spread_odds_dec,
                            stake=1.0, status=sp_status,
                            pnl=round(
                                (spread_odds_dec - 1.0) if sp_status == "won"
                                else (-1.0 if sp_status == "lost" else 0.0), 2),
                        ))
                        existing.add(key_sp)
                        inserted += 1

                # --- TOTALS ---
                total_line = _first_float(r, [
                    "total", "Total Score Close", "OU", "TotalLine",
                ])
                if total_line is not None:
                    actual_total = pts_h + pts_a
                    tot_odds = 1.91

                    # Over
                    tot_o_eid = _mk_event_id(["nba", game_id, date_str, home, away, f"o{total_line}"])
                    tot_o_sel = f"Over {total_line}"
                    tot_o_status = (
                        "won" if actual_total > total_line
                        else "lost" if actual_total < total_line
                        else "void"
                    )
                    key_to = (tot_o_eid, tot_o_sel)
                    if key_to not in existing:
                        db.add(PlacedBet(
                            event_id=tot_o_eid, sport=sport, market="totals",
                            selection=tot_o_sel, odds=tot_odds,
                            stake=1.0, status=tot_o_status,
                            pnl=round(
                                (tot_odds - 1.0) if tot_o_status == "won"
                                else (-1.0 if tot_o_status == "lost" else 0.0), 2),
                        ))
                        existing.add(key_to)
                        inserted += 1

                    # Under
                    tot_u_eid = _mk_event_id(["nba", game_id, date_str, home, away, f"u{total_line}"])
                    tot_u_sel = f"Under {total_line}"
                    tot_u_status = (
                        "won" if actual_total < total_line
                        else "lost" if actual_total > total_line
                        else "void"
                    )
                    key_tu = (tot_u_eid, tot_u_sel)
                    if key_tu not in existing:
                        db.add(PlacedBet(
                            event_id=tot_u_eid, sport=sport, market="totals",
                            selection=tot_u_sel, odds=tot_odds,
                            stake=1.0, status=tot_u_status,
                            pnl=round(
                                (tot_odds - 1.0) if tot_u_status == "won"
                                else (-1.0 if tot_u_status == "lost" else 0.0), 2),
                        ))
                        existing.add(key_tu)
                        inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# NFL — aussportsbetting format, decimal odds
# ---------------------------------------------------------------------------

def import_nfl(folder: Path, max_rows: int = 50000) -> int:
    """Import NFL game data (aussportsbetting format).

    Expected columns:
      Date, Home Team, Away Team, Home Score, Away Score,
      Home Odds Close, Away Odds Close,
      Home Line Close, Home Line Odds, Away Line Odds,
      Total Score Close, Total Score Over Odds, Total Score Under Odds
    All odds are decimal.
    """
    sport = csv_code_to_api_key("nfl") or "americanfootball_nfl"
    files = sorted(folder.glob("*.csv"))
    if not files:
        return 0

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue
            if max_rows > 0 and len(df) > max_rows:
                df = df.tail(max_rows)

            for _, r in df.iterrows():
                home = _first_str(r, [
                    "Home Team", "HomeTeam", "Home", "home_team",
                ]) or ""
                away = _first_str(r, [
                    "Away Team", "AwayTeam", "Away", "away_team",
                ]) or ""
                if not home or not away:
                    continue

                hs = _first_int(r, ["Home Score", "HomeScore", "Home Points", "PtsH"])
                a_s = _first_int(r, ["Away Score", "AwayScore", "Away Points", "PtsA"])
                if hs is None or a_s is None:
                    continue

                date_str = _first_str(r, ["Date", "date", "GameDate"]) or ""

                # --- MONEYLINE (decimal odds) ---
                home_ml = _first_num(r, [
                    "Home Odds Close", "Home Odds", "HomeOdds", "B365H", "PinnH",
                ])
                away_ml = _first_num(r, [
                    "Away Odds Close", "Away Odds", "AwayOdds", "B365A", "PinnA",
                ])

                if home_ml is not None:
                    ml_eid = _mk_event_id(["nfl", date_str, home, away, "ml_h"])
                    ml_status = "won" if hs > a_s else ("lost" if hs < a_s else "void")
                    key = (ml_eid, home)
                    if key not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid, sport=sport, market="h2h",
                            selection=home, odds=float(home_ml),
                            stake=1.0, status=ml_status,
                            pnl=round(
                                (home_ml - 1.0) if ml_status == "won"
                                else (-1.0 if ml_status == "lost" else 0.0), 2),
                        ))
                        existing.add(key)
                        inserted += 1

                if away_ml is not None:
                    ml_eid_a = _mk_event_id(["nfl", date_str, home, away, "ml_a"])
                    ml_status_a = "won" if a_s > hs else ("lost" if a_s < hs else "void")
                    key_a = (ml_eid_a, away)
                    if key_a not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid_a, sport=sport, market="h2h",
                            selection=away, odds=float(away_ml),
                            stake=1.0, status=ml_status_a,
                            pnl=round(
                                (away_ml - 1.0) if ml_status_a == "won"
                                else (-1.0 if ml_status_a == "lost" else 0.0), 2),
                        ))
                        existing.add(key_a)
                        inserted += 1

                # --- SPREADS ---
                spread_line = _first_float(r, [
                    "Home Line Close", "Home Line", "HomeLine", "HomeSpread",
                ])
                spread_h_odds = _first_num(r, [
                    "Home Line Odds", "Home Line Close Odds", "HomeLineOdds",
                ])
                if spread_line is not None and spread_h_odds is not None:
                    sp_eid = _mk_event_id(["nfl", date_str, home, away, "sp"])
                    home_adjusted = hs + spread_line
                    sp_status = (
                        "won" if home_adjusted > a_s
                        else "lost" if home_adjusted < a_s
                        else "void"
                    )
                    sp_sel = f"{home} {spread_line:+.1f}"
                    key_sp = (sp_eid, sp_sel)
                    if key_sp not in existing:
                        db.add(PlacedBet(
                            event_id=sp_eid, sport=sport, market="spreads",
                            selection=sp_sel, odds=float(spread_h_odds),
                            stake=1.0, status=sp_status,
                            pnl=round(
                                (spread_h_odds - 1.0) if sp_status == "won"
                                else (-1.0 if sp_status == "lost" else 0.0), 2),
                        ))
                        existing.add(key_sp)
                        inserted += 1

                # --- TOTALS ---
                total_line = _first_float(r, [
                    "Total Score Close", "Total Score Line", "TotalLine", "OU", "Total",
                ])
                over_odds = _first_num(r, [
                    "Total Score Over Odds", "Over Odds", "OverOdds",
                ])
                under_odds = _first_num(r, [
                    "Total Score Under Odds", "Under Odds", "UnderOdds",
                ])

                if total_line is not None and over_odds is not None:
                    actual_total = hs + a_s

                    tot_o_eid = _mk_event_id(["nfl", date_str, home, away, f"o{total_line}"])
                    tot_o_sel = f"Over {total_line}"
                    tot_o_status = (
                        "won" if actual_total > total_line
                        else "lost" if actual_total < total_line
                        else "void"
                    )
                    key_to = (tot_o_eid, tot_o_sel)
                    if key_to not in existing:
                        db.add(PlacedBet(
                            event_id=tot_o_eid, sport=sport, market="totals",
                            selection=tot_o_sel, odds=float(over_odds),
                            stake=1.0, status=tot_o_status,
                            pnl=round(
                                (over_odds - 1.0) if tot_o_status == "won"
                                else (-1.0 if tot_o_status == "lost" else 0.0), 2),
                        ))
                        existing.add(key_to)
                        inserted += 1

                    if under_odds is not None:
                        tot_u_eid = _mk_event_id(["nfl", date_str, home, away, f"u{total_line}"])
                        tot_u_sel = f"Under {total_line}"
                        tot_u_status = (
                            "won" if actual_total < total_line
                            else "lost" if actual_total > total_line
                            else "void"
                        )
                        key_tu = (tot_u_eid, tot_u_sel)
                        if key_tu not in existing:
                            db.add(PlacedBet(
                                event_id=tot_u_eid, sport=sport, market="totals",
                                selection=tot_u_sel, odds=float(under_odds),
                                stake=1.0, status=tot_u_status,
                                pnl=round(
                                    (under_odds - 1.0) if tot_u_status == "won"
                                    else (-1.0 if tot_u_status == "lost" else 0.0), 2),
                            ))
                            existing.add(key_tu)
                            inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# NHL — one-row-per-TEAM-per-game, American moneyline odds
# ---------------------------------------------------------------------------

def import_nhl(folder: Path, max_rows: int = 50000) -> int:
    """Import NHL game data.

    Many NHL datasets store one row per *team* per game (home row + away row
    share the same game_id). We group by game_id, then for each pair produce
    H2H, spreads and totals rows.

    Expected columns (common Kaggle NHL format):
      game_id, team, opponent, home_away (HOME/AWAY), goals, goals_against,
      favorite_moneyline (American odds), underdog_moneyline,
      spread, total, over_under_result
    """
    sport = csv_code_to_api_key("nhl") or "icehockey_nhl"
    files = sorted(folder.glob("*.csv"))
    if not files:
        return 0

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            df = _safe_read_csv(p)
            if df is None or df.empty:
                continue
            if max_rows > 0 and len(df) > max_rows:
                df = df.tail(max_rows)

            # Attempt to detect format
            cols_lower = {c.lower(): c for c in df.columns}
            has_home_away_col = "home_away" in cols_lower or "HoA" in df.columns

            if has_home_away_col:
                # --- ONE ROW PER TEAM FORMAT ---
                _import_nhl_per_team(df, sport, existing, db)
            else:
                # --- ONE ROW PER GAME FORMAT (aussportsbetting-like) ---
                _import_nhl_per_game(df, sport, existing, db)

            inserted = len(existing) - inserted  # approximate

        db.commit()

    # Re-count properly
    with SessionLocal() as db:
        total_nhl = db.query(func.count(PlacedBet.id)).filter(
            PlacedBet.sport == sport
        ).scalar() or 0
    return total_nhl


def _import_nhl_per_team(
    df: pd.DataFrame, sport: str,
    existing: set, db,
) -> None:
    """Parse NHL data where each row is one team's perspective of a game."""
    ha_col = "home_away" if "home_away" in df.columns else "HoA"

    # Group pairs by game_id
    gid_col = None
    for candidate in ["game_id", "Game ID", "gameId", "game_date"]:
        if candidate in df.columns:
            gid_col = candidate
            break
    if gid_col is None:
        return

    for gid, group in df.groupby(gid_col):
        home_rows = group[group[ha_col].astype(str).str.upper().str.startswith("H")]
        away_rows = group[group[ha_col].astype(str).str.upper().str.startswith("A")]
        if home_rows.empty or away_rows.empty:
            continue

        hr = home_rows.iloc[0]
        ar = away_rows.iloc[0]

        home = str(hr.get("team") or hr.get("Team") or "").strip()
        away = str(ar.get("team") or ar.get("Team") or "").strip()
        if not home or not away:
            continue

        goals_h = _first_int(hr, ["goals", "Goals", "GF", "goals_for"])
        goals_a = _first_int(ar, ["goals", "Goals", "GF", "goals_for"])
        if goals_h is None or goals_a is None:
            continue

        date_str = _first_str(hr, ["game_date", "Date", "date"]) or str(gid)

        # --- MONEYLINE ---
        # American odds: figure out who is favourite
        fav_ml_raw = _first_float(hr, ["favorite_moneyline", "Favorite ML", "FavoriteML"])
        und_ml_raw = _first_float(hr, ["underdog_moneyline", "Underdog ML", "UnderdogML"])

        # Some datasets just have team-specific ML columns
        home_ml_raw = _first_float(hr, [
            "moneyline", "ML", "team_moneyline", "Home ML",
        ]) or fav_ml_raw
        away_ml_raw = _first_float(ar, [
            "moneyline", "ML", "team_moneyline", "Away ML",
        ]) or und_ml_raw

        home_ml_dec = _american_to_decimal(home_ml_raw) if home_ml_raw is not None else None
        away_ml_dec = _american_to_decimal(away_ml_raw) if away_ml_raw is not None else None

        if home_ml_dec and home_ml_dec > 1.0:
            eid = _mk_event_id(["nhl", str(gid), date_str, home, away, "ml_h"])
            status = "won" if goals_h > goals_a else "lost"
            key = (eid, home)
            if key not in existing:
                db.add(PlacedBet(
                    event_id=eid, sport=sport, market="h2h",
                    selection=home, odds=float(home_ml_dec),
                    stake=1.0, status=status,
                    pnl=round((home_ml_dec - 1.0) if status == "won" else -1.0, 2),
                ))
                existing.add(key)

        if away_ml_dec and away_ml_dec > 1.0:
            eid_a = _mk_event_id(["nhl", str(gid), date_str, home, away, "ml_a"])
            status_a = "won" if goals_a > goals_h else "lost"
            key_a = (eid_a, away)
            if key_a not in existing:
                db.add(PlacedBet(
                    event_id=eid_a, sport=sport, market="h2h",
                    selection=away, odds=float(away_ml_dec),
                    stake=1.0, status=status_a,
                    pnl=round((away_ml_dec - 1.0) if status_a == "won" else -1.0, 2),
                ))
                existing.add(key_a)

        # --- TOTALS ---
        total_line = _first_float(hr, ["total", "Total", "OU", "over_under"])
        if total_line is not None:
            actual_total = goals_h + goals_a
            tot_eid = _mk_event_id(["nhl", str(gid), date_str, home, away, f"o{total_line}"])
            tot_sel = f"Over {total_line}"
            tot_status = (
                "won" if actual_total > total_line
                else "lost" if actual_total < total_line
                else "void"
            )
            key_t = (tot_eid, tot_sel)
            if key_t not in existing:
                db.add(PlacedBet(
                    event_id=tot_eid, sport=sport, market="totals",
                    selection=tot_sel, odds=1.91,
                    stake=1.0, status=tot_status,
                    pnl=round(0.91 if tot_status == "won"
                              else (-1.0 if tot_status == "lost" else 0.0), 2),
                ))
                existing.add(key_t)


def _import_nhl_per_game(
    df: pd.DataFrame, sport: str,
    existing: set, db,
) -> None:
    """Parse NHL data with one row per game (aussportsbetting-like)."""
    for _, r in df.iterrows():
        home = _first_str(r, ["Home Team", "HomeTeam", "Home", "home_team"]) or ""
        away = _first_str(r, ["Away Team", "AwayTeam", "Away", "away_team"]) or ""
        if not home or not away:
            continue

        hs = _first_int(r, ["Home Score", "HomeScore", "Home Goals", "PtsH"])
        a_s = _first_int(r, ["Away Score", "AwayScore", "Away Goals", "PtsA"])
        if hs is None or a_s is None:
            continue

        date_str = _first_str(r, ["Date", "date", "GameDate"]) or ""

        # Moneyline (decimal)
        home_ml = _first_num(r, ["Home Odds Close", "Home Odds", "HomeOdds", "B365H"])
        away_ml = _first_num(r, ["Away Odds Close", "Away Odds", "AwayOdds", "B365A"])

        if home_ml is not None:
            eid = _mk_event_id(["nhl", date_str, home, away, "ml_h"])
            status = "won" if hs > a_s else "lost"
            key = (eid, home)
            if key not in existing:
                db.add(PlacedBet(
                    event_id=eid, sport=sport, market="h2h",
                    selection=home, odds=float(home_ml),
                    stake=1.0, status=status,
                    pnl=round((home_ml - 1.0) if status == "won" else -1.0, 2),
                ))
                existing.add(key)

        if away_ml is not None:
            eid_a = _mk_event_id(["nhl", date_str, home, away, "ml_a"])
            status_a = "won" if a_s > hs else "lost"
            key_a = (eid_a, away)
            if key_a not in existing:
                db.add(PlacedBet(
                    event_id=eid_a, sport=sport, market="h2h",
                    selection=away, odds=float(away_ml),
                    stake=1.0, status=status_a,
                    pnl=round((away_ml - 1.0) if status_a == "won" else -1.0, 2),
                ))
                existing.add(key_a)

        # Spreads
        spread_line = _first_float(r, ["Home Line Close", "Home Line", "HomeSpread"])
        spread_odds = _first_num(r, ["Home Line Odds", "HomeLineOdds"])
        if spread_line is not None and spread_odds is not None:
            sp_eid = _mk_event_id(["nhl", date_str, home, away, "sp"])
            home_adj = hs + spread_line
            sp_status = (
                "won" if home_adj > a_s
                else "lost" if home_adj < a_s
                else "void"
            )
            sp_sel = f"{home} {spread_line:+.1f}"
            key_sp = (sp_eid, sp_sel)
            if key_sp not in existing:
                db.add(PlacedBet(
                    event_id=sp_eid, sport=sport, market="spreads",
                    selection=sp_sel, odds=float(spread_odds),
                    stake=1.0, status=sp_status,
                    pnl=round(
                        (spread_odds - 1.0) if sp_status == "won"
                        else (-1.0 if sp_status == "lost" else 0.0), 2),
                ))
                existing.add(key_sp)

        # Totals
        total_line = _first_float(r, ["Total Score Close", "TotalLine", "OU", "Total"])
        over_odds = _first_num(r, ["Total Score Over Odds", "Over Odds", "OverOdds"])
        under_odds = _first_num(r, ["Total Score Under Odds", "Under Odds", "UnderOdds"])
        if total_line is not None and over_odds is not None:
            actual_total = hs + a_s
            tot_o_eid = _mk_event_id(["nhl", date_str, home, away, f"o{total_line}"])
            tot_o_sel = f"Over {total_line}"
            tot_o_status = (
                "won" if actual_total > total_line
                else "lost" if actual_total < total_line
                else "void"
            )
            key_to = (tot_o_eid, tot_o_sel)
            if key_to not in existing:
                db.add(PlacedBet(
                    event_id=tot_o_eid, sport=sport, market="totals",
                    selection=tot_o_sel, odds=float(over_odds),
                    stake=1.0, status=tot_o_status,
                    pnl=round(
                        (over_odds - 1.0) if tot_o_status == "won"
                        else (-1.0 if tot_o_status == "lost" else 0.0), 2),
                ))
                existing.add(key_to)

            if under_odds is not None:
                tot_u_eid = _mk_event_id(["nhl", date_str, home, away, f"u{total_line}"])
                tot_u_sel = f"Under {total_line}"
                tot_u_status = (
                    "won" if actual_total < total_line
                    else "lost" if actual_total > total_line
                    else "void"
                )
                key_tu = (tot_u_eid, tot_u_sel)
                if key_tu not in existing:
                    db.add(PlacedBet(
                        event_id=tot_u_eid, sport=sport, market="totals",
                        selection=tot_u_sel, odds=float(under_odds),
                        stake=1.0, status=tot_u_status,
                        pnl=round(
                            (under_odds - 1.0) if tot_u_status == "won"
                            else (-1.0 if tot_u_status == "lost" else 0.0), 2),
                    ))
                    existing.add(key_tu)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Import historical results (football, tennis, NBA, NFL, NHL) into placed_bets"
    )
    ap.add_argument("--imports-dir", default="data/imports")
    ap.add_argument("--football-dir", type=str, default=None, help="football CSV dir (overrides --imports-dir/football)")
    ap.add_argument("--football-files", type=int, default=0, help="limit football CSV files (0=all)")
    ap.add_argument("--tennis-dir", type=str, default=None, help="tennis file dir (overrides --imports-dir/tennis)")
    ap.add_argument("--tennis-files", type=int, default=0, help="limit tennis files (0=all)")
    ap.add_argument("--nba-dir", type=str, default=None, help="path to NBA CSV directory")
    ap.add_argument("--nfl-dir", type=str, default=None, help="path to NFL CSV directory")
    ap.add_argument("--nhl-dir", type=str, default=None, help="path to NHL CSV directory")
    ap.add_argument("--max-rows", type=int, default=50000, help="max rows per US sport (0=all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    Base.metadata.create_all(bind=engine)

    base = Path(args.imports_dir)
    counts = {}

    # Football
    fb_dir = Path(args.football_dir) if args.football_dir else base / "football"
    if fb_dir.exists():
        n = import_football(fb_dir, limit_files=args.football_files)
        counts["football"] = n
        log.info("Football: %d rows inserted", n)
    else:
        counts["football"] = 0

    # Tennis
    tn_dir = Path(args.tennis_dir) if args.tennis_dir else base / "tennis"
    if tn_dir.exists():
        n = import_tennis(tn_dir, limit_files=args.tennis_files)
        counts["tennis"] = n
        log.info("Tennis: %d rows inserted", n)
    else:
        counts["tennis"] = 0

    # NBA
    nba_dir = Path(args.nba_dir) if args.nba_dir else base / "nba"
    if nba_dir.exists():
        n = import_nba(nba_dir, max_rows=args.max_rows)
        counts["nba"] = n
        log.info("NBA: %d rows inserted", n)
    else:
        counts["nba"] = 0

    # NFL
    nfl_dir = Path(args.nfl_dir) if args.nfl_dir else base / "nfl"
    if nfl_dir.exists():
        n = import_nfl(nfl_dir, max_rows=args.max_rows)
        counts["nfl"] = n
        log.info("NFL: %d rows inserted", n)
    else:
        counts["nfl"] = 0

    # NHL
    nhl_dir = Path(args.nhl_dir) if args.nhl_dir else base / "nhl"
    if nhl_dir.exists():
        n = import_nhl(nhl_dir, max_rows=args.max_rows)
        counts["nhl"] = n
        log.info("NHL: %d rows inserted", n)
    else:
        counts["nhl"] = 0

    with SessionLocal() as db:
        total = db.scalar(select(func.count()).select_from(PlacedBet))
    counts["total_after"] = total
    print(counts)


if __name__ == "__main__":
    main()
