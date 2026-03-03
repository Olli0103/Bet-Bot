#!/usr/bin/env python3
"""Intelligent Universal Data Importer for historical sports results.

Extracts MAXIMUM ML value from CSV files:
  - Standard columns (odds, status, pnl) in the PlacedBet schema
  - Advanced sport-specific stats in the ``meta_features`` JSONB column

Supports:
  - Football (soccer): H2H, Totals, BTTS + shots, corners, cards, AHC
  - Tennis: ATP/WTA/Challenger H2H + rankings, surface, sets
  - NBA: H2H, Spreads, Totals + quarter scores, playoffs, American odds
  - NFL: H2H, Spreads, Totals + playoff flags, line momentum
  - NHL: H2H, Totals + shots, power play, faceoffs, American odds

All sync DB calls are safe because this runs as a standalone CLI script.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import select, func, text

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


def _safe_float(val: Any) -> Optional[float]:
    """Coerce *val* to float, returning None on failure or NaN."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if not math.isnan(f) else None
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Coerce *val* to int, returning None on failure."""
    f = _safe_float(val)
    return int(f) if f is not None else None


def _first_num(row, cols: list[str]) -> Optional[float]:
    """Return first numeric value > 1.0 from candidate column names (for odds)."""
    for c in cols:
        if c in row.index:
            v = _safe_float(row[c])
            if v is not None and v > 1.0:
                return v
    return None


def _first_float(row, cols: list[str]) -> Optional[float]:
    """Return first valid float from candidate column names."""
    for c in cols:
        if c in row.index:
            v = _safe_float(row[c])
            if v is not None:
                return v
    return None


def _first_int(row, cols: list[str]) -> Optional[int]:
    """Return first valid integer from candidate column names."""
    for c in cols:
        if c in row.index:
            v = _safe_int(row[c])
            if v is not None:
                return v
    return None


def _first_str(row, cols: list[str]) -> Optional[str]:
    """Return first non-empty string value from candidate column names."""
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            val = str(row[c]).strip()
            if val:
                return val
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


def _clean_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    """Strip None values and convert numpy types for JSONB storage."""
    out = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        # numpy scalar -> python native
        if hasattr(v, "item"):
            v = v.item()
        out[k] = v
    return out


def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    """Read a CSV with lenient error handling + numeric coercion."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", **kwargs)
            return df
        except Exception:
            continue
    log.warning("Cannot read %s with any encoding", path)
    return None


def _pnl(odds: float, status: str) -> float:
    if status == "won":
        return round(odds - 1.0, 2)
    elif status == "lost":
        return -1.0
    return 0.0


def _ensure_schema() -> None:
    """Add any columns present in the ORM model but missing from the DB table.

    This resolves the ``meta_features column missing`` error that occurs when
    the ``placed_bets`` table was originally created from an older model that
    lacked the JSONB column.  Uses ``ADD COLUMN IF NOT EXISTS`` so it is safe
    to call on every run.
    """
    migrations = [
        "ALTER TABLE placed_bets ADD COLUMN IF NOT EXISTS meta_features JSONB",
        "ALTER TABLE placed_bets ADD COLUMN IF NOT EXISTS odds_open DOUBLE PRECISION",
        "ALTER TABLE placed_bets ADD COLUMN IF NOT EXISTS odds_close DOUBLE PRECISION",
        "ALTER TABLE placed_bets ADD COLUMN IF NOT EXISTS clv DOUBLE PRECISION DEFAULT 0.0",
        "ALTER TABLE placed_bets ADD COLUMN IF NOT EXISTS form_winrate_l5 DOUBLE PRECISION",
        "ALTER TABLE placed_bets ADD COLUMN IF NOT EXISTS form_games_l5 DOUBLE PRECISION",
    ]
    with engine.begin() as conn:
        for stmt in migrations:
            conn.execute(text(stmt))
    log.info("Schema check passed – all required columns present.")


# ---------------------------------------------------------------------------
# Football (Soccer): H2H + Totals + BTTS
#   Deep extraction: half-time, shots, corners, fouls, cards, AHC
# ---------------------------------------------------------------------------

def import_football(folder: Path, limit_files: int = 0) -> int:
    """Import football CSVs with deep feature extraction.

    Uses ``Div`` column mapped through SPORT_MAPPING for correct Odds API keys.
    Extracts advanced stats (shots, corners, cards, AHC lines) into meta_features.
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
            # Coerce known numeric columns once upfront
            num_cols = [c for c in df.columns if c in {
                "FTHG", "FTAG", "HTHG", "HTAG",
                "HS", "AS", "HST", "AST", "HC", "AC",
                "HF", "AF", "HY", "AY", "HR", "AR",
                "B365H", "B365D", "B365A", "B365>2.5", "B365<2.5",
                "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
                "AvgH", "AvgD", "AvgA", "Avg>2.5", "Avg<2.5",
                "AHh", "AHCh", "PAHH", "PAHA",
            }]
            for c in num_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            for _, r in df.iterrows():
                home = str(r.get("HomeTeam") or r.get("HT") or "").strip()
                away = str(r.get("AwayTeam") or r.get("AT") or "").strip()
                ftr = str(r.get("FTR") or "").upper().strip()
                if not home or not away or ftr not in {"H", "D", "A"}:
                    continue

                # --- Resolve sport key via central mapping ---
                div_raw = str(r.get("Div") or "").strip()
                sport = csv_code_to_api_key(div_raw) or f"soccer_{div_raw.lower()}"

                base_eid = _mk_event_id(["football", div_raw, r.get("Date"), home, away])

                # --- Build shared meta_features dict ---
                fthg = _safe_int(r.get("FTHG"))
                ftag = _safe_int(r.get("FTAG"))

                meta: Dict[str, Any] = _clean_meta({
                    # Half-time
                    "hthg": _safe_int(r.get("HTHG")),
                    "htag": _safe_int(r.get("HTAG")),
                    "htr": _first_str(r, ["HTR"]),
                    # Shots
                    "home_shots": _safe_int(r.get("HS")),
                    "away_shots": _safe_int(r.get("AS")),
                    "home_shots_target": _safe_int(r.get("HST")),
                    "away_shots_target": _safe_int(r.get("AST")),
                    # Corners
                    "home_corners": _safe_int(r.get("HC")),
                    "away_corners": _safe_int(r.get("AC")),
                    # Fouls
                    "home_fouls": _safe_int(r.get("HF")),
                    "away_fouls": _safe_int(r.get("AF")),
                    # Cards
                    "home_yellows": _safe_int(r.get("HY")),
                    "away_yellows": _safe_int(r.get("AY")),
                    "home_reds": _safe_int(r.get("HR")),
                    "away_reds": _safe_int(r.get("AR")),
                    # Asian Handicap line (if available)
                    "ahc_line": _safe_float(r.get("AHh") or r.get("AHCh")),
                    "ahc_home_odds": _safe_float(r.get("PAHH")),
                    "ahc_away_odds": _safe_float(r.get("PAHA")),
                })

                # --- H2H market ---
                oh_open = _first_num(r, ["B365H", "WHH", "VCH", "IWH", "LBH"])
                od_open = _first_num(r, ["B365D", "WHD", "VCD", "IWD", "LBD"])
                oa_open = _first_num(r, ["B365A", "WHA", "VCA", "IWA", "LBA"])

                oh_close = _first_num(r, ["PSH", "PSCH", "AvgH", "MaxH", "B365H"])
                od_close = _first_num(r, ["PSD", "PSCD", "AvgD", "MaxD", "B365D"])
                oa_close = _first_num(r, ["PSA", "PSCA", "AvgA", "MaxA", "B365A"])

                cand = [("H", oh_close), ("D", od_close), ("A", oa_close)]
                cand = [(k, v) for k, v in cand if v is not None]
                if cand:
                    pick, odds_close = sorted(cand, key=lambda x: x[1])[0]
                    selection = home if pick == "H" else away if pick == "A" else "Draw"
                    status = "won" if pick == ftr else "lost"
                    odds_open_map = {"H": oh_open, "D": od_open, "A": oa_open}
                    odds_open = odds_open_map.get(pick) or odds_close

                    key = (base_eid, selection)
                    if key not in existing:
                        db.add(PlacedBet(
                            event_id=base_eid, sport=sport, market="h2h",
                            selection=selection, odds=float(odds_close),
                            odds_open=float(odds_open), odds_close=float(odds_close),
                            clv=round((float(odds_open) / float(odds_close)) - 1.0, 4)
                                if float(odds_close) > 1.0 else 0.0,
                            stake=1.0, status=status, pnl=_pnl(odds_close, status),
                            meta_features=meta,
                        ))
                        existing.add(key)
                        inserted += 1

                # --- Totals + BTTS ---
                if fthg is not None and ftag is not None:
                    total_goals = fthg + ftag

                    # Over/Under 2.5
                    over_25 = _first_num(r, ["B365>2.5", "P>2.5", "BbAv>2.5", "Avg>2.5", "Max>2.5"]) or 1.85
                    under_25 = _first_num(r, ["B365<2.5", "P<2.5", "BbAv<2.5", "Avg<2.5", "Max<2.5"]) or 1.95

                    for sel, threshold, odds_val, suffix in [
                        ("Over 2.5", 2, over_25, "o25"),
                        ("Under 2.5", 2, under_25, "u25"),
                    ]:
                        is_over = sel.startswith("Over")
                        st = ("won" if total_goals > threshold else "lost") if is_over \
                            else ("won" if total_goals <= threshold else "lost")
                        eid = _mk_event_id(["football", div_raw, r.get("Date"), home, away, suffix])
                        k = (eid, sel)
                        if k not in existing:
                            db.add(PlacedBet(
                                event_id=eid, sport=sport, market="totals",
                                selection=sel, odds=odds_val,
                                stake=1.0, status=st, pnl=_pnl(odds_val, st),
                                meta_features=meta,
                            ))
                            existing.add(k)
                            inserted += 1

                    # Over 1.5 (derive if no explicit column)
                    over_15 = _first_num(r, ["B365>1.5", "Avg>1.5", "Max>1.5"])
                    if over_15 is None and over_25 > 1.0:
                        ip_25 = 1.0 / over_25
                        ip_15 = min(0.95, ip_25 + 0.20)
                        over_15 = round(1.0 / ip_15, 2) if ip_15 > 0.01 else None

                    if over_15 and over_15 > 1.0:
                        st_15 = "won" if total_goals > 1 else "lost"
                        eid_15 = _mk_event_id(["football", div_raw, r.get("Date"), home, away, "o15"])
                        k15 = (eid_15, "Over 1.5")
                        if k15 not in existing:
                            db.add(PlacedBet(
                                event_id=eid_15, sport=sport, market="totals",
                                selection="Over 1.5", odds=over_15,
                                stake=1.0, status=st_15, pnl=_pnl(over_15, st_15),
                                meta_features=meta,
                            ))
                            existing.add(k15)
                            inserted += 1

                    # BTTS
                    btts_odds = _first_num(r, ["BbAvBTTS", "AvgBTTS", "MaxBTTS"]) or 1.75
                    st_btts = "won" if fthg >= 1 and ftag >= 1 else "lost"
                    eid_btts = _mk_event_id(["football", div_raw, r.get("Date"), home, away, "btts"])
                    k_btts = (eid_btts, "BTTS Yes")
                    if k_btts not in existing:
                        db.add(PlacedBet(
                            event_id=eid_btts, sport=sport, market="btts",
                            selection="BTTS Yes", odds=btts_odds,
                            stake=1.0, status=st_btts, pnl=_pnl(btts_odds, st_btts),
                            meta_features=meta,
                        ))
                        existing.add(k_btts)
                        inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# Tennis: ATP, WTA, Challenger
#   Deep extraction: rankings, points, surface, sets
# ---------------------------------------------------------------------------

def import_tennis(folder: Path, limit_files: int = 0) -> int:
    """Import tennis files with ranking/surface extraction.

    Generates Winner AND Loser rows so ML sees both sides of every match.
    Extracts WRank, LRank, WPts, LPts, Surface, sets into meta_features.
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

            # Coerce ranking/points columns
            for c in ["WRank", "LRank", "WPts", "LPts", "W1", "L1", "W2", "L2",
                       "W3", "L3", "W4", "L4", "W5", "L5", "Wsets", "Lsets"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            # Filter out incomplete matches (Retired, Walkover, Default,
            # Disqualified) to prevent ML data poisoning.  Only rows
            # with Comment == "Completed" represent true match outcomes.
            if "Comment" in df.columns:
                before = len(df)
                df = df[df["Comment"].astype(str).str.strip().str.lower() == "completed"]
                dropped = before - len(df)
                if dropped > 0:
                    log.info("Tennis %s: filtered %d incomplete matches (Retired/Walkover/etc.)", p.name, dropped)

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

                # --- Shared meta_features ---
                w_rank = _safe_int(r.get("WRank"))
                l_rank = _safe_int(r.get("LRank"))
                w_pts = _safe_int(r.get("WPts"))
                l_pts = _safe_int(r.get("LPts"))

                base_meta = _clean_meta({
                    "winner_rank": w_rank,
                    "loser_rank": l_rank,
                    "winner_points": w_pts,
                    "loser_points": l_pts,
                    "rank_diff": (w_rank - l_rank) if (w_rank and l_rank) else None,
                    "surface": _first_str(r, ["Surface", "surface"]),
                    "tournament": _first_str(r, ["Tournament", "tournament"]),
                    "round": _first_str(r, ["Round", "round"]),
                    "best_of": _safe_int(r.get("Best of") or r.get("best_of")),
                    "w_sets": _safe_int(r.get("Wsets")),
                    "l_sets": _safe_int(r.get("Lsets")),
                    # Set scores for set-by-set analysis
                    "w1": _safe_int(r.get("W1")), "l1": _safe_int(r.get("L1")),
                    "w2": _safe_int(r.get("W2")), "l2": _safe_int(r.get("L2")),
                    "w3": _safe_int(r.get("W3")), "l3": _safe_int(r.get("L3")),
                    "w4": _safe_int(r.get("W4")), "l4": _safe_int(r.get("L4")),
                    "w5": _safe_int(r.get("W5")), "l5": _safe_int(r.get("L5")),
                })

                # --- Winner row (status=won) ---
                if w_close is not None:
                    w_odds_open = w_open or w_close
                    w_meta = dict(base_meta)
                    w_meta["is_favourite"] = bool(w_close <= (l_close or 99.0))
                    w_meta["selection_rank"] = w_rank
                    w_meta["opponent_rank"] = l_rank
                    w_clv = round((w_odds_open / w_close) - 1.0, 4) if w_close > 1.0 else 0.0

                    key_w = (event_id, winner)
                    if key_w not in existing:
                        db.add(PlacedBet(
                            event_id=event_id, sport=sport, market="h2h",
                            selection=winner, odds=float(w_close),
                            odds_open=float(w_odds_open), odds_close=float(w_close),
                            clv=w_clv, stake=1.0, status="won",
                            pnl=round(w_close - 1.0, 2),
                            meta_features=_clean_meta(w_meta),
                        ))
                        existing.add(key_w)
                        inserted += 1

                # --- Loser row (status=lost) ---
                if l_close is not None:
                    l_odds_open = l_open or l_close
                    l_meta = dict(base_meta)
                    l_meta["is_favourite"] = bool(l_close <= (w_close or 99.0))
                    l_meta["selection_rank"] = l_rank
                    l_meta["opponent_rank"] = w_rank
                    l_clv = round((l_odds_open / l_close) - 1.0, 4) if l_close > 1.0 else 0.0

                    key_l = (event_id, loser)
                    if key_l not in existing:
                        db.add(PlacedBet(
                            event_id=event_id, sport=sport, market="h2h",
                            selection=loser, odds=float(l_close),
                            odds_open=float(l_odds_open), odds_close=float(l_close),
                            clv=l_clv, stake=1.0, status="lost", pnl=-1.0,
                            meta_features=_clean_meta(l_meta),
                        ))
                        existing.add(key_l)
                        inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# NBA — Kaggle one-row-per-game, American moneyline odds
#   Deep extraction: quarter scores, playoffs, favourite indicator
# ---------------------------------------------------------------------------

def import_nba(folder: Path, max_rows: int = 50000) -> int:
    """Import NBA data with quarter-by-quarter scoring and playoff flags."""
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

            # Coerce numeric columns
            for c in df.columns:
                if any(pat in c.lower() for pat in ["pts", "score", "moneyline", "spread", "total",
                                                     "q1", "q2", "q3", "q4", "ot"]):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            for _, r in df.iterrows():
                home = _first_str(r, ["team_home", "Home Team", "HomeTeam", "Home", "home_team", "home"]) or ""
                away = _first_str(r, ["team_away", "Away Team", "AwayTeam", "Away", "away_team", "away"]) or ""
                if not home or not away:
                    continue

                pts_h = _first_int(r, ["pts_home", "Home Score", "HomeScore", "PtsH", "Home Points", "score_home"])
                pts_a = _first_int(r, ["pts_away", "Away Score", "AwayScore", "PtsA", "Away Points", "score_away"])
                if pts_h is None or pts_a is None:
                    continue

                date_str = _first_str(r, ["game_date", "Date", "date", "GameDate"]) or ""
                game_id = _first_str(r, ["game_id", "Game ID", "gameId", "ID"]) or ""

                # --- meta_features: quarter scores, playoffs, favourite ---
                meta = _clean_meta({
                    "playoffs": _first_str(r, ["playoffs", "Playoff", "is_playoff"]),
                    "whos_favored": _first_str(r, ["whos_favored", "Favored", "favourite"]),
                    "q1_home": _first_int(r, ["q1_home", "Home Q1", "HQ1"]),
                    "q1_away": _first_int(r, ["q1_away", "Away Q1", "AQ1"]),
                    "q2_home": _first_int(r, ["q2_home", "Home Q2", "HQ2"]),
                    "q2_away": _first_int(r, ["q2_away", "Away Q2", "AQ2"]),
                    "q3_home": _first_int(r, ["q3_home", "Home Q3", "HQ3"]),
                    "q3_away": _first_int(r, ["q3_away", "Away Q3", "AQ3"]),
                    "q4_home": _first_int(r, ["q4_home", "Home Q4", "HQ4"]),
                    "q4_away": _first_int(r, ["q4_away", "Away Q4", "AQ4"]),
                    "ot_home": _first_int(r, ["ot_home", "Home OT"]),
                    "ot_away": _first_int(r, ["ot_away", "Away OT"]),
                })

                # --- MONEYLINE (H2H) — American odds ---
                ml_home_raw = _first_float(r, ["moneyline_home", "Home ML", "HomeML"])
                ml_away_raw = _first_float(r, ["moneyline_away", "Away ML", "AwayML"])

                home_ml_dec = _american_to_decimal(ml_home_raw) if ml_home_raw is not None else None
                away_ml_dec = _american_to_decimal(ml_away_raw) if ml_away_raw is not None else None

                # Fallback: try decimal columns
                if home_ml_dec is None:
                    home_ml_dec = _first_num(r, ["Home Odds Close", "Home Odds", "PinnH", "B365H"])
                if away_ml_dec is None:
                    away_ml_dec = _first_num(r, ["Away Odds Close", "Away Odds", "PinnA", "B365A"])

                if home_ml_dec and home_ml_dec > 1.0:
                    ml_eid = _mk_event_id(["nba", game_id, date_str, home, away, "ml_h"])
                    st = "won" if pts_h > pts_a else "lost"
                    k = (ml_eid, home)
                    if k not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid, sport=sport, market="h2h",
                            selection=home, odds=float(home_ml_dec),
                            stake=1.0, status=st, pnl=_pnl(home_ml_dec, st),
                            meta_features=meta,
                        ))
                        existing.add(k)
                        inserted += 1

                if away_ml_dec and away_ml_dec > 1.0:
                    ml_eid_a = _mk_event_id(["nba", game_id, date_str, home, away, "ml_a"])
                    st_a = "won" if pts_a > pts_h else "lost"
                    k_a = (ml_eid_a, away)
                    if k_a not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid_a, sport=sport, market="h2h",
                            selection=away, odds=float(away_ml_dec),
                            stake=1.0, status=st_a, pnl=_pnl(away_ml_dec, st_a),
                            meta_features=meta,
                        ))
                        existing.add(k_a)
                        inserted += 1

                # --- SPREADS ---
                spread_raw = _first_float(r, ["spread", "Home Line Close", "HomeSpread", "Home Line"])
                spread_odds = 1.91  # NBA standard vig
                if spread_raw is not None:
                    sp_eid = _mk_event_id(["nba", game_id, date_str, home, away, "sp"])
                    home_adj = pts_h + spread_raw
                    sp_st = "won" if home_adj > pts_a else ("lost" if home_adj < pts_a else "void")
                    sp_sel = f"{home} {spread_raw:+.1f}"
                    k_sp = (sp_eid, sp_sel)
                    if k_sp not in existing:
                        db.add(PlacedBet(
                            event_id=sp_eid, sport=sport, market="spreads",
                            selection=sp_sel, odds=spread_odds,
                            stake=1.0, status=sp_st, pnl=_pnl(spread_odds, sp_st),
                            meta_features=meta,
                        ))
                        existing.add(k_sp)
                        inserted += 1

                # --- TOTALS ---
                total_line = _first_float(r, ["total", "Total Score Close", "OU", "TotalLine"])
                if total_line is not None:
                    actual_total = pts_h + pts_a
                    tot_odds = 1.91
                    for sel_prefix, cmp_fn, suffix in [
                        ("Over", lambda a, l: a > l, "o"),
                        ("Under", lambda a, l: a < l, "u"),
                    ]:
                        tot_eid = _mk_event_id(["nba", game_id, date_str, home, away, f"{suffix}{total_line}"])
                        tot_sel = f"{sel_prefix} {total_line}"
                        tot_st = "won" if cmp_fn(actual_total, total_line) else ("lost" if actual_total != total_line else "void")
                        k_t = (tot_eid, tot_sel)
                        if k_t not in existing:
                            db.add(PlacedBet(
                                event_id=tot_eid, sport=sport, market="totals",
                                selection=tot_sel, odds=tot_odds,
                                stake=1.0, status=tot_st, pnl=_pnl(tot_odds, tot_st),
                                meta_features=meta,
                            ))
                            existing.add(k_t)
                            inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# NFL — aussportsbetting format, decimal odds
#   Deep extraction: playoff flag, neutral venue, line momentum
# ---------------------------------------------------------------------------

def import_nfl(folder: Path, max_rows: int = 50000) -> int:
    """Import NFL data with momentum (open vs close) and playoff flags."""
    sport = csv_code_to_api_key("nfl") or "americanfootball_nfl"
    csv_files = sorted(folder.glob("*.csv"))
    xlsx_files = sorted(folder.glob("*.xlsx"))
    xls_files = sorted(folder.glob("*.xls"))
    files = csv_files + xlsx_files + xls_files
    if not files:
        return 0

    inserted = 0
    with SessionLocal() as db:
        existing = set(db.query(PlacedBet.event_id, PlacedBet.selection).all())

        for p in files:
            if p.suffix == ".csv":
                df = _safe_read_csv(p)
            else:
                try:
                    df = pd.read_excel(p)
                except Exception:
                    log.warning("Cannot read Excel file %s", p)
                    continue
            if df is None or df.empty:
                continue
            if max_rows > 0 and len(df) > max_rows:
                df = df.tail(max_rows)

            # Coerce numerics
            for c in df.columns:
                if any(pat in c.lower() for pat in ["odds", "score", "line", "total"]):
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            for _, r in df.iterrows():
                home = _first_str(r, ["Home Team", "HomeTeam", "Home", "home_team"]) or ""
                away = _first_str(r, ["Away Team", "AwayTeam", "Away", "away_team"]) or ""
                if not home or not away:
                    continue

                hs = _first_int(r, ["Home Score", "HomeScore", "Home Points", "PtsH"])
                a_s = _first_int(r, ["Away Score", "AwayScore", "Away Points", "PtsA"])
                if hs is None or a_s is None:
                    continue

                date_str = _first_str(r, ["Date", "date", "GameDate"]) or ""

                # --- meta_features: momentum + flags ---
                h_odds_open = _first_float(r, ["Home Odds Open", "HomeOddsOpen"])
                h_odds_close = _first_float(r, ["Home Odds Close", "Home Odds", "HomeOdds"])
                total_open = _first_float(r, ["Total Score Open", "TotalOpen"])
                total_close = _first_float(r, ["Total Score Close", "TotalClose"])

                momentum_h2h = None
                if h_odds_open and h_odds_close and h_odds_open > 1 and h_odds_close > 1:
                    momentum_h2h = round((1.0 / h_odds_close) - (1.0 / h_odds_open), 4)

                momentum_total = None
                if total_open and total_close:
                    momentum_total = round(total_close - total_open, 2)

                meta = _clean_meta({
                    "is_playoff": _first_str(r, ["Playoff Game?", "Playoff", "is_playoff"]),
                    "neutral_venue": _first_str(r, ["Neutral Venue?", "NeutralVenue"]),
                    "momentum_h2h": momentum_h2h,
                    "momentum_total": momentum_total,
                    "home_odds_open": h_odds_open,
                    "home_odds_close": h_odds_close,
                })

                # --- MONEYLINE (decimal odds) ---
                home_ml = _first_num(r, ["Home Odds Close", "Home Odds", "HomeOdds", "B365H", "PinnH"])
                away_ml = _first_num(r, ["Away Odds Close", "Away Odds", "AwayOdds", "B365A", "PinnA"])

                if home_ml is not None:
                    ml_eid = _mk_event_id(["nfl", date_str, home, away, "ml_h"])
                    st = "won" if hs > a_s else ("lost" if hs < a_s else "void")
                    k = (ml_eid, home)
                    if k not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid, sport=sport, market="h2h",
                            selection=home, odds=float(home_ml),
                            stake=1.0, status=st, pnl=_pnl(home_ml, st),
                            meta_features=meta,
                        ))
                        existing.add(k)
                        inserted += 1

                if away_ml is not None:
                    ml_eid_a = _mk_event_id(["nfl", date_str, home, away, "ml_a"])
                    st_a = "won" if a_s > hs else ("lost" if a_s < hs else "void")
                    k_a = (ml_eid_a, away)
                    if k_a not in existing:
                        db.add(PlacedBet(
                            event_id=ml_eid_a, sport=sport, market="h2h",
                            selection=away, odds=float(away_ml),
                            stake=1.0, status=st_a, pnl=_pnl(away_ml, st_a),
                            meta_features=meta,
                        ))
                        existing.add(k_a)
                        inserted += 1

                # --- SPREADS ---
                spread_line = _first_float(r, ["Home Line Close", "Home Line", "HomeLine", "HomeSpread"])
                spread_h_odds = _first_num(r, ["Home Line Odds", "Home Line Close Odds", "HomeLineOdds"])
                if spread_line is not None and spread_h_odds is not None:
                    sp_eid = _mk_event_id(["nfl", date_str, home, away, "sp"])
                    home_adj = hs + spread_line
                    sp_st = "won" if home_adj > a_s else ("lost" if home_adj < a_s else "void")
                    sp_sel = f"{home} {spread_line:+.1f}"
                    k_sp = (sp_eid, sp_sel)
                    if k_sp not in existing:
                        db.add(PlacedBet(
                            event_id=sp_eid, sport=sport, market="spreads",
                            selection=sp_sel, odds=float(spread_h_odds),
                            stake=1.0, status=sp_st, pnl=_pnl(spread_h_odds, sp_st),
                            meta_features=meta,
                        ))
                        existing.add(k_sp)
                        inserted += 1

                # --- TOTALS ---
                total_line_val = _first_float(r, ["Total Score Close", "Total Score Line", "TotalLine", "OU", "Total"])
                over_odds = _first_num(r, ["Total Score Over Odds", "Over Odds", "OverOdds"])
                under_odds = _first_num(r, ["Total Score Under Odds", "Under Odds", "UnderOdds"])

                if total_line_val is not None and over_odds is not None:
                    actual_total = hs + a_s
                    for sel_pfx, odds_v, cmp_fn, sfx in [
                        ("Over", over_odds, lambda a, l: a > l, "o"),
                        ("Under", under_odds, lambda a, l: a < l, "u"),
                    ]:
                        if odds_v is None:
                            continue
                        tot_eid = _mk_event_id(["nfl", date_str, home, away, f"{sfx}{total_line_val}"])
                        tot_sel = f"{sel_pfx} {total_line_val}"
                        tot_st = "won" if cmp_fn(actual_total, total_line_val) else (
                            "lost" if actual_total != total_line_val else "void")
                        k_t = (tot_eid, tot_sel)
                        if k_t not in existing:
                            db.add(PlacedBet(
                                event_id=tot_eid, sport=sport, market="totals",
                                selection=tot_sel, odds=float(odds_v),
                                stake=1.0, status=tot_st, pnl=_pnl(odds_v, tot_st),
                                meta_features=meta,
                            ))
                            existing.add(k_t)
                            inserted += 1

        db.commit()
    return inserted


# ---------------------------------------------------------------------------
# NHL — one-row-per-team or one-row-per-game, American moneyline odds
#   Deep extraction: shots, power play, faceoffs, hits, PIM, rolling stats
# ---------------------------------------------------------------------------

def import_nhl(folder: Path, max_rows: int = 50000) -> int:
    """Import NHL data with advanced stats (shots, PP, faceoffs, rolling averages)."""
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

            # Detect format
            has_home_away_col = any(c in df.columns for c in ["home_away", "HoA", "Home/Away"])
            has_is_home_col = "is_home" in df.columns

            if has_is_home_col:
                n = _import_nhl_is_home(df, sport, existing, db)
            elif has_home_away_col:
                n = _import_nhl_per_team(df, sport, existing, db)
            else:
                n = _import_nhl_per_game(df, sport, existing, db)
            inserted += n

        db.commit()
    return inserted


def _extract_nhl_meta(row, prefix: str = "") -> Dict[str, Any]:
    """Extract NHL advanced stats from a row into a meta dict."""
    pfx = f"{prefix}_" if prefix else ""
    return _clean_meta({
        f"{pfx}shots": _safe_int(row.get("shots") or row.get("Shots")),
        f"{pfx}power_play_goals": _safe_int(row.get("powerPlayGoals") or row.get("PPG")),
        f"{pfx}power_play_opps": _safe_int(row.get("powerPlayOpportunities") or row.get("PPO")),
        f"{pfx}faceoff_win_pct": _safe_float(row.get("faceOffWinPercentage") or row.get("FO%")),
        f"{pfx}hits": _safe_int(row.get("hits") or row.get("Hits")),
        f"{pfx}blocked_shots": _safe_int(row.get("blocked") or row.get("Blocked")),
        f"{pfx}pim": _safe_int(row.get("pim") or row.get("PIM")),
        f"{pfx}takeaways": _safe_int(row.get("takeaways") or row.get("Takeaways")),
        f"{pfx}giveaways": _safe_int(row.get("giveaways") or row.get("Giveaways")),
        f"{pfx}rest_days": _safe_int(row.get("rest_days") or row.get("DaysRest")),
        # Rolling averages
        f"{pfx}roll_3_goals": _safe_float(row.get("roll_3_goals")),
        f"{pfx}roll_3_shots": _safe_float(row.get("roll_3_shots")),
        f"{pfx}roll_3_save_pct": _safe_float(row.get("roll_3_save_pct")),
        f"{pfx}roll_10_goals": _safe_float(row.get("roll_10_goals")),
        f"{pfx}roll_10_shots": _safe_float(row.get("roll_10_shots")),
        f"{pfx}roll_10_save_pct": _safe_float(row.get("roll_10_save_pct")),
    })


def _import_nhl_per_team(
    df: pd.DataFrame, sport: str, existing: set, db,
) -> int:
    """Parse NHL data where each row is one team's perspective of a game."""
    inserted = 0
    ha_col = next((c for c in ["home_away", "HoA", "Home/Away"] if c in df.columns), None)
    if ha_col is None:
        return 0

    gid_col = next((c for c in ["game_id", "Game ID", "gameId", "game_date"] if c in df.columns), None)
    if gid_col is None:
        return 0

    # Coerce numerics
    for c in df.columns:
        if any(pat in c.lower() for pat in ["goals", "shots", "hit", "block", "pim",
                                             "faceoff", "roll_", "moneyline", "spread", "total"]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

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

        goals_h = _safe_int(hr.get("goals") or hr.get("Goals") or hr.get("GF"))
        goals_a = _safe_int(ar.get("goals") or ar.get("Goals") or ar.get("GF"))
        if goals_h is None or goals_a is None:
            continue

        date_str = _first_str(hr, ["game_date", "Date", "date"]) or str(gid)

        # --- Build rich meta ---
        meta_home = _extract_nhl_meta(hr, "home")
        meta_away = _extract_nhl_meta(ar, "away")
        meta = {**meta_home, **meta_away}

        # --- MONEYLINE (American odds) ---
        fav_ml = _safe_float(hr.get("favorite_moneyline") or hr.get("Favorite ML"))
        und_ml = _safe_float(hr.get("underdog_moneyline") or hr.get("Underdog ML"))
        home_ml_raw = _safe_float(hr.get("moneyline") or hr.get("ML") or hr.get("team_moneyline")) or fav_ml
        away_ml_raw = _safe_float(ar.get("moneyline") or ar.get("ML") or ar.get("team_moneyline")) or und_ml

        home_ml_dec = _american_to_decimal(home_ml_raw) if home_ml_raw else None
        away_ml_dec = _american_to_decimal(away_ml_raw) if away_ml_raw else None

        if home_ml_dec and home_ml_dec > 1.0:
            eid = _mk_event_id(["nhl", str(gid), date_str, home, away, "ml_h"])
            st = "won" if goals_h > goals_a else "lost"
            k = (eid, home)
            if k not in existing:
                db.add(PlacedBet(
                    event_id=eid, sport=sport, market="h2h",
                    selection=home, odds=float(home_ml_dec),
                    stake=1.0, status=st, pnl=_pnl(home_ml_dec, st),
                    meta_features=meta,
                ))
                existing.add(k)
                inserted += 1

        if away_ml_dec and away_ml_dec > 1.0:
            eid_a = _mk_event_id(["nhl", str(gid), date_str, home, away, "ml_a"])
            st_a = "won" if goals_a > goals_h else "lost"
            k_a = (eid_a, away)
            if k_a not in existing:
                db.add(PlacedBet(
                    event_id=eid_a, sport=sport, market="h2h",
                    selection=away, odds=float(away_ml_dec),
                    stake=1.0, status=st_a, pnl=_pnl(away_ml_dec, st_a),
                    meta_features=meta,
                ))
                existing.add(k_a)
                inserted += 1

        # --- TOTALS ---
        total_line = _safe_float(hr.get("total") or hr.get("Total") or hr.get("OU"))
        if total_line is not None:
            actual_total = goals_h + goals_a
            tot_eid = _mk_event_id(["nhl", str(gid), date_str, home, away, f"o{total_line}"])
            tot_st = "won" if actual_total > total_line else ("lost" if actual_total < total_line else "void")
            k_t = (tot_eid, f"Over {total_line}")
            if k_t not in existing:
                db.add(PlacedBet(
                    event_id=tot_eid, sport=sport, market="totals",
                    selection=f"Over {total_line}", odds=1.91,
                    stake=1.0, status=tot_st, pnl=_pnl(1.91, tot_st),
                    meta_features=meta,
                ))
                existing.add(k_t)
                inserted += 1

    return inserted


def _import_nhl_is_home(
    df: pd.DataFrame, sport: str, existing: set, db,
) -> int:
    """Parse NHL data in team-perspective format with ``is_home`` boolean.

    Expected columns: ``game_id``, ``is_home``, ``team_name``, ``opp_team_name``,
    ``goals_for``, ``goals_against``, and optionally moneyline / totals odds.
    Each game appears as two rows (one per team).  We group by ``game_id``
    and pick the home row to build a canonical per-game record.
    """
    inserted = 0

    gid_col = next(
        (c for c in ["game_id", "Game ID", "gameId"] if c in df.columns), None
    )
    if gid_col is None:
        log.warning("NHL is_home format detected but no game_id column found")
        return 0

    # Coerce numerics
    for c in df.columns:
        if any(pat in c.lower() for pat in [
            "goals", "shots", "hit", "block", "pim",
            "faceoff", "roll_", "moneyline", "spread", "total",
        ]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for gid, group in df.groupby(gid_col):
        # Normalise is_home to boolean
        home_mask = group["is_home"].astype(str).str.strip().str.lower().isin(
            ["1", "true", "yes", "1.0"]
        )
        away_mask = ~home_mask

        home_rows = group[home_mask]
        away_rows = group[away_mask]
        if home_rows.empty or away_rows.empty:
            continue

        hr = home_rows.iloc[0]
        ar = away_rows.iloc[0]

        home = str(hr.get("team_name") or hr.get("team") or "").strip()
        away = str(ar.get("team_name") or ar.get("team") or "").strip()
        if not home or not away:
            continue

        goals_h = _safe_int(hr.get("goals_for") or hr.get("GF") or hr.get("goals"))
        goals_a = _safe_int(ar.get("goals_for") or ar.get("GF") or ar.get("goals"))
        if goals_h is None or goals_a is None:
            # Fallback: try goals_against from the opposite row
            if goals_h is None:
                goals_h = _safe_int(ar.get("goals_against") or ar.get("GA"))
            if goals_a is None:
                goals_a = _safe_int(hr.get("goals_against") or hr.get("GA"))
        if goals_h is None or goals_a is None:
            continue

        date_str = _first_str(hr, ["game_date", "Date", "date"]) or str(gid)

        # --- Build meta from both rows ---
        meta_home = _extract_nhl_meta(hr, "home")
        meta_away = _extract_nhl_meta(ar, "away")
        meta = {**meta_home, **meta_away}

        # --- MONEYLINE (American odds -> decimal) ---
        home_ml_raw = _safe_float(
            hr.get("moneyline") or hr.get("ML") or hr.get("team_moneyline")
        )
        away_ml_raw = _safe_float(
            ar.get("moneyline") or ar.get("ML") or ar.get("team_moneyline")
        )

        home_ml_dec = _american_to_decimal(home_ml_raw) if home_ml_raw else None
        away_ml_dec = _american_to_decimal(away_ml_raw) if away_ml_raw else None

        if home_ml_dec and home_ml_dec > 1.0:
            eid = _mk_event_id(["nhl", str(gid), date_str, home, away, "ml_h"])
            st = "won" if goals_h > goals_a else "lost"
            k = (eid, home)
            if k not in existing:
                db.add(PlacedBet(
                    event_id=eid, sport=sport, market="h2h",
                    selection=home, odds=float(home_ml_dec),
                    stake=1.0, status=st, pnl=_pnl(home_ml_dec, st),
                    meta_features=meta,
                ))
                existing.add(k)
                inserted += 1

        if away_ml_dec and away_ml_dec > 1.0:
            eid_a = _mk_event_id(["nhl", str(gid), date_str, home, away, "ml_a"])
            st_a = "won" if goals_a > goals_h else "lost"
            k_a = (eid_a, away)
            if k_a not in existing:
                db.add(PlacedBet(
                    event_id=eid_a, sport=sport, market="h2h",
                    selection=away, odds=float(away_ml_dec),
                    stake=1.0, status=st_a, pnl=_pnl(away_ml_dec, st_a),
                    meta_features=meta,
                ))
                existing.add(k_a)
                inserted += 1

        # --- TOTALS (Over) ---
        total_line = _safe_float(
            hr.get("total") or hr.get("Total") or hr.get("OU")
        )
        if total_line is not None:
            actual_total = goals_h + goals_a
            tot_eid = _mk_event_id(["nhl", str(gid), date_str, home, away, f"o{total_line}"])
            tot_st = "won" if actual_total > total_line else (
                "lost" if actual_total < total_line else "void"
            )
            k_t = (tot_eid, f"Over {total_line}")
            if k_t not in existing:
                db.add(PlacedBet(
                    event_id=tot_eid, sport=sport, market="totals",
                    selection=f"Over {total_line}", odds=1.91,
                    stake=1.0, status=tot_st, pnl=_pnl(1.91, tot_st),
                    meta_features=meta,
                ))
                existing.add(k_t)
                inserted += 1

    return inserted


def _import_nhl_per_game(
    df: pd.DataFrame, sport: str, existing: set, db,
) -> int:
    """Parse NHL data with one row per game (aussportsbetting-like)."""
    inserted = 0

    for c in df.columns:
        if any(pat in c.lower() for pat in ["odds", "score", "line", "total"]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
        meta: Dict[str, Any] = {}

        # Moneyline (decimal)
        home_ml = _first_num(r, ["Home Odds Close", "Home Odds", "HomeOdds", "B365H"])
        away_ml = _first_num(r, ["Away Odds Close", "Away Odds", "AwayOdds", "B365A"])

        if home_ml is not None:
            eid = _mk_event_id(["nhl", date_str, home, away, "ml_h"])
            st = "won" if hs > a_s else "lost"
            k = (eid, home)
            if k not in existing:
                db.add(PlacedBet(
                    event_id=eid, sport=sport, market="h2h",
                    selection=home, odds=float(home_ml),
                    stake=1.0, status=st, pnl=_pnl(home_ml, st),
                    meta_features=meta,
                ))
                existing.add(k)
                inserted += 1

        if away_ml is not None:
            eid_a = _mk_event_id(["nhl", date_str, home, away, "ml_a"])
            st_a = "won" if a_s > hs else "lost"
            k_a = (eid_a, away)
            if k_a not in existing:
                db.add(PlacedBet(
                    event_id=eid_a, sport=sport, market="h2h",
                    selection=away, odds=float(away_ml),
                    stake=1.0, status=st_a, pnl=_pnl(away_ml, st_a),
                    meta_features=meta,
                ))
                existing.add(k_a)
                inserted += 1

        # Spreads
        spread_line = _first_float(r, ["Home Line Close", "Home Line", "HomeSpread"])
        spread_odds = _first_num(r, ["Home Line Odds", "HomeLineOdds"])
        if spread_line is not None and spread_odds is not None:
            sp_eid = _mk_event_id(["nhl", date_str, home, away, "sp"])
            home_adj = hs + spread_line
            sp_st = "won" if home_adj > a_s else ("lost" if home_adj < a_s else "void")
            sp_sel = f"{home} {spread_line:+.1f}"
            k_sp = (sp_eid, sp_sel)
            if k_sp not in existing:
                db.add(PlacedBet(
                    event_id=sp_eid, sport=sport, market="spreads",
                    selection=sp_sel, odds=float(spread_odds),
                    stake=1.0, status=sp_st, pnl=_pnl(spread_odds, sp_st),
                    meta_features=meta,
                ))
                existing.add(k_sp)
                inserted += 1

        # Totals
        total_line = _first_float(r, ["Total Score Close", "TotalLine", "OU", "Total"])
        over_odds = _first_num(r, ["Total Score Over Odds", "Over Odds", "OverOdds"])
        under_odds = _first_num(r, ["Total Score Under Odds", "Under Odds", "UnderOdds"])
        if total_line is not None and over_odds is not None:
            actual_total = hs + a_s
            for sel_pfx, odds_v, cmp_fn, sfx in [
                ("Over", over_odds, lambda a, l: a > l, "o"),
                ("Under", under_odds, lambda a, l: a < l, "u"),
            ]:
                if odds_v is None:
                    continue
                tot_eid = _mk_event_id(["nhl", date_str, home, away, f"{sfx}{total_line}"])
                tot_sel = f"{sel_pfx} {total_line}"
                tot_st = "won" if cmp_fn(actual_total, total_line) else (
                    "lost" if actual_total != total_line else "void")
                k_t = (tot_eid, tot_sel)
                if k_t not in existing:
                    db.add(PlacedBet(
                        event_id=tot_eid, sport=sport, market="totals",
                        selection=tot_sel, odds=float(odds_v),
                        stake=1.0, status=tot_st, pnl=_pnl(odds_v, tot_st),
                        meta_features=meta,
                    ))
                    existing.add(k_t)
                    inserted += 1

    return inserted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Import historical results (football, tennis, NBA, NFL, NHL) into placed_bets"
    )
    ap.add_argument("--imports-dir", default="data/imports")
    ap.add_argument("--football-dir", type=str, default=None)
    ap.add_argument("--football-files", type=int, default=0, help="limit football CSV files (0=all)")
    ap.add_argument("--tennis-dir", type=str, default=None)
    ap.add_argument("--tennis-files", type=int, default=0, help="limit tennis files (0=all)")
    ap.add_argument("--nba-dir", type=str, default=None)
    ap.add_argument("--nfl-dir", type=str, default=None)
    ap.add_argument("--nhl-dir", type=str, default=None)
    ap.add_argument("--max-rows", type=int, default=50000, help="max rows per US sport file (0=all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    Base.metadata.create_all(bind=engine)
    _ensure_schema()

    base = Path(args.imports_dir)
    counts: Dict[str, int] = {}

    # Football
    fb_dir = Path(args.football_dir) if args.football_dir else base / "football"
    if fb_dir.exists():
        n = import_football(fb_dir, limit_files=args.football_files)
        counts["football"] = n
        log.info("Football: %d rows inserted", n)

    # Tennis
    tn_dir = Path(args.tennis_dir) if args.tennis_dir else base / "tennis"
    if tn_dir.exists():
        n = import_tennis(tn_dir, limit_files=args.tennis_files)
        counts["tennis"] = n
        log.info("Tennis: %d rows inserted", n)

    # NBA
    nba_dir = Path(args.nba_dir) if args.nba_dir else base / "nba"
    if nba_dir.exists():
        n = import_nba(nba_dir, max_rows=args.max_rows)
        counts["nba"] = n
        log.info("NBA: %d rows inserted", n)

    # NFL
    nfl_dir = Path(args.nfl_dir) if args.nfl_dir else base / "nfl"
    if nfl_dir.exists():
        n = import_nfl(nfl_dir, max_rows=args.max_rows)
        counts["nfl"] = n
        log.info("NFL: %d rows inserted", n)

    # NHL
    nhl_dir = Path(args.nhl_dir) if args.nhl_dir else base / "nhl"
    if nhl_dir.exists():
        n = import_nhl(nhl_dir, max_rows=args.max_rows)
        counts["nhl"] = n
        log.info("NHL: %d rows inserted", n)

    with SessionLocal() as db:
        total = db.scalar(select(func.count()).select_from(PlacedBet))
    counts["total_after"] = total or 0
    print(counts)


if __name__ == "__main__":
    main()
