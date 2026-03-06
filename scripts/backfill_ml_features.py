#!/usr/bin/env python3
"""Backfill missing ML features in placed_bets.meta_features.

Phase 1 (critical):
- sentiment_delta  -> 0.0 (neutral fallback)
- injury_delta     -> 0.0 (neutral fallback)
- sharp_implied_prob / sharp_vig -> derived from odds, else neutral
- form_winrate_l5 / form_games_l5 -> from TeamMatchStats history, else neutral

Phase 2-4 (enrichment + stats):
- elo_diff, elo_expected, h2h_home_winrate, home_advantage -> neutral defaults
- weather_rain, weather_wind_high -> 0.0 (assume dry)
- home_volatility, away_volatility, is_steam_move, line_staleness -> 0.0
- injury_news_delta, time_to_kickoff_hours, public_bias -> neutral
- market_momentum, line_velocity -> 0.0
- Stats features (attack/defense strength, form_trend_slope, etc.) -> league avg

Batchwise, idempotent, supports --dry-run, --limit, --sport, --force.
Prints coverage before/after for verification.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from math import isfinite
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Ensure project root is importable
sys.path.insert(0, ".")

from src.data.models import PlacedBet, TeamMatchStats  # noqa: E402
from src.data.postgres import SessionLocal  # noqa: E402

CRITICAL_FEATURES = [
    "sentiment_delta",
    "injury_delta",
    "sharp_implied_prob",
    "sharp_vig",
    "form_winrate_l5",
    "form_games_l5",
]

# Phase 2-4 features with their semantic neutral defaults.
# These are the features that were 100% NaN in training because the backfill
# script only covered Phase 1 critical features.
PHASE2_DEFAULTS: Dict[str, float] = {
    "elo_diff": 0.0,
    "elo_expected": 0.5,
    "h2h_home_winrate": 0.5,
    "home_advantage": 0.5,
    "weather_rain": 0.0,
    "weather_wind_high": 0.0,
    "home_volatility": 0.0,
    "away_volatility": 0.0,
    "is_steam_move": 0.0,
    "line_staleness": 0.0,
    "injury_news_delta": 0.0,
    "time_to_kickoff_hours": 24.0,
    "public_bias": 0.0,
    "market_momentum": 0.0,
    "line_velocity": 0.0,
}

PHASE4_DEFAULTS: Dict[str, float] = {
    "team_attack_strength": 1.0,
    "team_defense_strength": 1.0,
    "opp_attack_strength": 1.0,
    "opp_defense_strength": 1.0,
    "expected_total_proxy": 2.7,
    "form_trend_slope": 0.0,
    "rest_fatigue_score": 0.3,
    "schedule_congestion": 0.15,
    "over25_rate": 0.55,
    "btts_rate": 0.50,
    "home_away_split_delta": 0.0,
    "league_position_delta": 0.0,
    "goals_scored_avg": 1.35,
    "goals_conceded_avg": 1.35,
}

ALL_BACKFILL_FEATURES = CRITICAL_FEATURES + list(PHASE2_DEFAULTS.keys()) + list(PHASE4_DEFAULTS.keys())

BATCH_SIZE = 200


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _derive_sharp_implied_prob(odds: float, vig: float) -> float:
    """Derive sharp implied probability from odds with vig correction."""
    if not isfinite(odds) or odds <= 1.0:
        return 0.5
    raw = 1.0 / odds
    overround = 1.0 + vig
    if overround > 0:
        return _clip(raw / overround, 0.01, 0.99)
    return _clip(raw, 0.01, 0.99)


def _derive_sharp_vig(odds: float) -> float:
    """Estimate vig from a single odds value (assumes ~5% market standard)."""
    if not isfinite(odds) or odds <= 1.0:
        return 0.05
    return 0.05  # conservative default


def _true_vig_from_prices(prices: Dict) -> Optional[float]:
    vals: List[float] = []
    for v in (prices or {}).values():
        try:
            vf = float(v)
        except Exception:
            continue
        if isfinite(vf) and vf > 1.0:
            vals.append(vf)
    if len(vals) < 2:
        return None
    implied_sum = sum(1.0 / x for x in vals)
    vig = implied_sum - 1.0
    return round(max(0.001, vig), 4)


def _derive_sharp_vig_from_meta(meta: Dict, odds: float) -> Tuple[float, str]:
    """Prefer true market-book overround from meta_features when available."""
    prices = meta.get("sharp_prices_h2h")
    if isinstance(prices, dict):
        true_vig = _true_vig_from_prices(prices)
        if true_vig is not None:
            return true_vig, "true_book"

    # Secondary: explicit close odds pairs sometimes stored in meta
    home_close = meta.get("home_odds_close")
    away_close = meta.get("away_odds_close")
    if home_close and away_close:
        true_vig = _true_vig_from_prices({"home": home_close, "away": away_close})
        if true_vig is not None:
            return true_vig, "paired_close"

    return _derive_sharp_vig(odds), "heuristic_fallback"


def _compute_form_from_history(
    team: str,
    match_date=None,
    window: int = 5,
) -> Tuple[float, int]:
    """Compute form_winrate_l5 and form_games_l5 from TeamMatchStats."""
    try:
        with SessionLocal() as db:
            stmt = select(TeamMatchStats).where(
                TeamMatchStats.team == team
            ).order_by(TeamMatchStats.match_date.desc()).limit(window)

            if match_date:
                stmt = select(TeamMatchStats).where(
                    TeamMatchStats.team == team,
                    TeamMatchStats.match_date < match_date,
                ).order_by(TeamMatchStats.match_date.desc()).limit(window)

            rows = db.scalars(stmt).all()
            if not rows:
                return 0.5, 0
            wins = sum(1 for r in rows if r.result == "W")
            total = len(rows)
            return round(wins / max(1, total), 4), total
    except Exception:
        return 0.5, 0


def _ensure_meta_features(bet: PlacedBet) -> dict:
    """Return the meta_features dict, creating it if None."""
    if bet.meta_features and isinstance(bet.meta_features, dict):
        return dict(bet.meta_features)
    return {}


def _needs_backfill(meta: dict, force: bool = False) -> bool:
    """Check if any feature (critical or phase 2-4) is missing."""
    if force:
        return True
    if not meta:
        return True
    for feat in ALL_BACKFILL_FEATURES:
        if meta.get(feat) is None:
            return True
    return False


def _extract_team_from_selection(selection: str) -> str:
    """Extract team name from selection string like 'Team (Home vs Away)'."""
    if " (" in selection:
        return selection.split(" (")[0].strip()
    return selection.strip()


def _compute_coverage(bets: list, features: List[str]) -> Dict[str, float]:
    """Compute non-null rate for each feature across all bets."""
    total = len(bets)
    if total == 0:
        return {f: 0.0 for f in features}

    coverage: Dict[str, int] = {f: 0 for f in features}
    for bet in bets:
        meta = bet.meta_features if isinstance(bet.meta_features, dict) else {}
        for f in features:
            if meta.get(f) is not None:
                coverage[f] += 1
    return {f: round(c / total, 4) for f, c in coverage.items()}


def backfill(
    limit: int = 0,
    sport_filter: Optional[str] = None,
    dry_run: bool = False,
    force: bool = False,
) -> Dict:
    """Backfill missing ML features in placed_bets.

    Returns statistics dict with coverage before/after.
    """
    stats: Dict = {
        "scanned": 0,
        "updated": 0,
        "skipped": 0,
        "features_filled": {f: 0 for f in ALL_BACKFILL_FEATURES},
        "missing_flags": 0,
        "coverage_before": {},
        "coverage_after": {},
    }

    with SessionLocal() as db:
        stmt = select(PlacedBet).order_by(PlacedBet.created_at.asc())
        if sport_filter:
            stmt = stmt.where(PlacedBet.sport.ilike(f"%{sport_filter}%"))

        rows = db.scalars(stmt).all()
        total_rows = len(rows)
        log.info("Found %d rows to scan%s", total_rows, f" (sport={sport_filter})" if sport_filter else "")

        # Coverage BEFORE backfill
        stats["coverage_before"] = _compute_coverage(rows, ALL_BACKFILL_FEATURES)
        log.info("Coverage BEFORE backfill:")
        for feat, rate in stats["coverage_before"].items():
            if rate < 1.0:
                log.info("  %s: %.1f%%", feat, rate * 100)

        batch_count = 0
        for bet in rows:
            if limit and stats["scanned"] >= limit:
                break
            stats["scanned"] += 1

            meta = _ensure_meta_features(bet)

            if not _needs_backfill(meta, force):
                stats["skipped"] += 1
                continue

            odds = float(bet.odds or 2.0)
            odds_open = float(bet.odds_open or odds)
            odds_close = float(bet.odds_close or odds)
            best_odds = max(odds, odds_open, odds_close)

            changed = False

            # --- Phase 1: Critical features ---
            # sentiment_delta
            if meta.get("sentiment_delta") is None:
                val = float(bet.sentiment_delta) if bet.sentiment_delta is not None else 0.0
                meta["sentiment_delta"] = val
                stats["features_filled"]["sentiment_delta"] += 1
                changed = True

            # injury_delta
            if meta.get("injury_delta") is None:
                val = float(bet.injury_delta) if bet.injury_delta is not None else 0.0
                meta["injury_delta"] = val
                stats["features_filled"]["injury_delta"] += 1
                changed = True

            # sharp_vig
            if meta.get("sharp_vig") is None:
                if bet.sharp_vig is not None and bet.sharp_vig != 0.0:
                    val = float(bet.sharp_vig)
                    method = "column_existing"
                else:
                    val, method = _derive_sharp_vig_from_meta(meta, best_odds)
                meta["sharp_vig"] = val
                meta["_sharp_vig_method"] = method
                stats["features_filled"]["sharp_vig"] += 1
                changed = True

            # sharp_implied_prob
            if meta.get("sharp_implied_prob") is None:
                val = float(bet.sharp_implied_prob) if bet.sharp_implied_prob is not None and bet.sharp_implied_prob > 0.0 else _derive_sharp_implied_prob(best_odds, meta.get("sharp_vig", 0.05))
                meta["sharp_implied_prob"] = val
                if bet.sharp_implied_prob is None or bet.sharp_implied_prob == 0.0:
                    meta["_sharp_implied_prob_derived"] = True
                    stats["missing_flags"] += 1
                stats["features_filled"]["sharp_implied_prob"] += 1
                changed = True

            # form_winrate_l5 / form_games_l5
            if meta.get("form_winrate_l5") is None or meta.get("form_games_l5") is None:
                team = _extract_team_from_selection(bet.selection)
                wr, gp = _compute_form_from_history(team, bet.created_at)

                if meta.get("form_winrate_l5") is None:
                    col_val = float(bet.form_winrate_l5) if bet.form_winrate_l5 is not None else None
                    if col_val is not None and col_val != 0.5:
                        meta["form_winrate_l5"] = col_val
                    elif gp > 0:
                        meta["form_winrate_l5"] = wr
                    else:
                        meta["form_winrate_l5"] = 0.5
                    stats["features_filled"]["form_winrate_l5"] += 1
                    changed = True

                if meta.get("form_games_l5") is None:
                    col_val = float(bet.form_games_l5) if bet.form_games_l5 is not None else None
                    if col_val is not None and col_val > 0:
                        meta["form_games_l5"] = col_val
                    elif gp > 0:
                        meta["form_games_l5"] = float(gp)
                    else:
                        meta["form_games_l5"] = 0.0
                    stats["features_filled"]["form_games_l5"] += 1
                    changed = True

            # --- Phase 2: Market + enrichment features ---
            # Determine home_advantage from selection vs event teams
            is_home_sel = None
            if meta.get("home_advantage") is None:
                # Best-effort: check if selection matches home_team
                # (already stored in meta from live pipeline or notes)
                is_home_sel = 0.5  # unknown
                meta["home_advantage"] = is_home_sel
                stats["features_filled"]["home_advantage"] += 1
                changed = True

            for feat, default in PHASE2_DEFAULTS.items():
                if feat == "home_advantage":
                    continue  # handled above
                if meta.get(feat) is None:
                    meta[feat] = default
                    stats["features_filled"][feat] += 1
                    changed = True

            # --- Phase 4: Stats-based features ---
            for feat, default in PHASE4_DEFAULTS.items():
                if meta.get(feat) is None:
                    meta[feat] = default
                    stats["features_filled"][feat] += 1
                    changed = True

            if changed:
                # Sync dedicated columns from meta
                bet.sentiment_delta = float(meta.get("sentiment_delta", 0.0))
                bet.injury_delta = float(meta.get("injury_delta", 0.0))
                bet.sharp_implied_prob = float(meta.get("sharp_implied_prob", 0.0))
                bet.sharp_vig = float(meta.get("sharp_vig", 0.0))
                bet.form_winrate_l5 = float(meta.get("form_winrate_l5", 0.5))
                bet.form_games_l5 = float(meta.get("form_games_l5", 0.0))
                bet.meta_features = meta
                stats["updated"] += 1
                batch_count += 1

            # Commit in batches
            if batch_count >= BATCH_SIZE:
                if not dry_run:
                    db.commit()
                    log.info("Committed batch: %d rows updated so far", stats["updated"])
                batch_count = 0

        # Final commit
        if not dry_run and batch_count > 0:
            db.commit()

        # Coverage AFTER backfill
        # Re-read meta_features for coverage calculation
        stats["coverage_after"] = _compute_coverage(rows, ALL_BACKFILL_FEATURES)

    return stats


def main():
    ap = argparse.ArgumentParser(
        description="Backfill missing ML features in placed_bets.meta_features"
    )
    ap.add_argument("--limit", type=int, default=0, help="Max rows to process (0=all)")
    ap.add_argument("--sport", type=str, default=None, help="Filter by sport (e.g. soccer, tennis)")
    ap.add_argument("--dry-run", action="store_true", help="Preview changes without writing to DB")
    ap.add_argument("--force", action="store_true", help="Rewrite all rows, not only missing")
    args = ap.parse_args()

    log.info(
        "Starting backfill: limit=%s sport=%s dry_run=%s force=%s",
        args.limit or "all", args.sport or "all", args.dry_run, args.force,
    )

    result = backfill(
        limit=args.limit,
        sport_filter=args.sport,
        dry_run=args.dry_run,
        force=args.force,
    )

    log.info("Backfill complete:")
    log.info("  Scanned: %d", result["scanned"])
    log.info("  Updated: %d", result["updated"])
    log.info("  Skipped: %d", result["skipped"])
    log.info("  Missing flags added: %d", result["missing_flags"])
    log.info("  Features filled:")
    for feat, count in result["features_filled"].items():
        if count > 0:
            log.info("    %s: %d", feat, count)

    log.info("  Coverage BEFORE -> AFTER:")
    for feat in ALL_BACKFILL_FEATURES:
        before = result["coverage_before"].get(feat, 0.0)
        after = result["coverage_after"].get(feat, 0.0)
        if before < 1.0 or after < 1.0:
            log.info("    %s: %.1f%% -> %.1f%%", feat, before * 100, after * 100)

    if args.dry_run:
        log.info("  (DRY RUN — no changes written)")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
