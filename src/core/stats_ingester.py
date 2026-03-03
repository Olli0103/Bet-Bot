"""Stats ingestion pipeline — fetches match results and computes rolling features.

Orchestrates TheSportsDB + football-data.org data into TeamMatchStats rows,
then computes EventStatsSnapshot features for upcoming events.
Designed to run periodically (every 6h via APScheduler or manually).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select, and_

from src.core.settings import settings
from src.data.models import TeamMatchStats, EventStatsSnapshot
from src.data.postgres import SessionLocal

log = logging.getLogger(__name__)

ROLLING_WINDOW = 10  # number of recent matches to consider
LEAGUE_AVG_GOALS = 1.35  # fallback league average per team per match


def _normalize_team(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


# --------------------------------------------------------------------------
# 1. Ingest match results from data sources
# --------------------------------------------------------------------------

def ingest_from_sportsdb(sport_keys: Optional[List[str]] = None) -> int:
    """Fetch and store past events from TheSportsDB."""
    from src.integrations.sportsdb_fetcher import SportsDBFetcher

    fetcher = SportsDBFetcher(api_key=settings.sportsdb_api_key)
    if not sport_keys:
        sport_keys = [s.strip() for s in settings.live_sports.split(",") if s.strip()]

    inserted = 0
    for sport_key in sport_keys:
        try:
            events = fetcher.get_past_events(sport_key, rounds=10)
        except Exception as exc:
            log.warning("SportsDB ingest failed for %s: %s", sport_key, exc)
            continue

        with SessionLocal() as db:
            for event in events:
                if event["home_score"] is None or event["away_score"] is None:
                    continue
                inserted += _upsert_match_stats(
                    db, event, sport_key, source="thesportsdb"
                )
            db.commit()

    log.info("SportsDB ingestion: %d new rows from %d sports", inserted, len(sport_keys))
    return inserted


def ingest_from_football_data(sport_keys: Optional[List[str]] = None) -> int:
    """Fetch and store recent match results from football-data.org."""
    from src.integrations.football_data_fetcher import FootballDataFetcher

    if not settings.football_data_api_key:
        log.info("football-data.org API key not set, skipping ingestion")
        return 0

    fetcher = FootballDataFetcher(api_key=settings.football_data_api_key)
    if not sport_keys:
        sport_keys = [s.strip() for s in settings.live_sports.split(",")
                      if s.strip().startswith("soccer")]

    inserted = 0
    for sport_key in sport_keys:
        try:
            matches = fetcher.get_matches(sport_key, status="FINISHED", limit=50)
        except Exception as exc:
            log.warning("football-data.org ingest failed for %s: %s", sport_key, exc)
            continue

        with SessionLocal() as db:
            for match in matches:
                if match["home_score"] is None or match["away_score"] is None:
                    continue
                inserted += _upsert_match_stats_fdata(
                    db, match, sport_key, source="football-data.org"
                )
            db.commit()

    log.info("football-data.org ingestion: %d new rows from %d sports", inserted, len(sport_keys))
    return inserted


def _upsert_match_stats(db, event: Dict[str, Any], sport_key: str, source: str) -> int:
    """Insert home + away TeamMatchStats from a SportsDB event. Returns count of new rows."""
    match_id = str(event.get("event_id", ""))
    if not match_id:
        return 0

    home = event["home_team"]
    away = event["away_team"]
    home_score = event["home_score"]
    away_score = event["away_score"]

    match_date_str = event.get("date", "")
    try:
        match_date = datetime.fromisoformat(match_date_str).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        match_date = datetime.now(timezone.utc)

    def _result(gf: int, ga: int) -> str:
        if gf > ga:
            return "W"
        elif gf == ga:
            return "D"
        return "L"

    count = 0
    for team, opponent, is_home, gf, ga in [
        (home, away, True, home_score, away_score),
        (away, home, False, away_score, home_score),
    ]:
        existing = db.execute(
            select(TeamMatchStats).where(
                and_(
                    TeamMatchStats.source_match_id == match_id,
                    TeamMatchStats.team == team,
                    TeamMatchStats.source == source,
                )
            )
        ).scalar_one_or_none()

        if existing:
            continue

        row = TeamMatchStats(
            source_match_id=match_id,
            sport=sport_key,
            league=event.get("league", ""),
            season=event.get("season", ""),
            matchday=int(event.get("round", 0) or 0),
            match_date=match_date,
            team=team,
            opponent=opponent,
            is_home=is_home,
            goals_for=gf,
            goals_against=ga,
            result=_result(gf, ga),
            shots=event.get("home_shots") if is_home else event.get("away_shots"),
            ht_goals_for=None,
            ht_goals_against=None,
            source=source,
        )
        db.add(row)
        count += 1

    return count


def _upsert_match_stats_fdata(db, match: Dict[str, Any], sport_key: str, source: str) -> int:
    """Insert home + away TeamMatchStats from a football-data.org match."""
    match_id = str(match.get("match_id", ""))
    if not match_id:
        return 0

    home = match["home_team"]
    away = match["away_team"]
    home_score = match["home_score"]
    away_score = match["away_score"]

    date_str = match.get("date", "")
    try:
        match_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        match_date = datetime.now(timezone.utc)

    def _result(gf: int, ga: int) -> str:
        if gf > ga:
            return "W"
        elif gf == ga:
            return "D"
        return "L"

    count = 0
    for team, opponent, is_home, gf, ga in [
        (home, away, True, home_score, away_score),
        (away, home, False, away_score, home_score),
    ]:
        existing = db.execute(
            select(TeamMatchStats).where(
                and_(
                    TeamMatchStats.source_match_id == match_id,
                    TeamMatchStats.team == team,
                    TeamMatchStats.source == source,
                )
            )
        ).scalar_one_or_none()

        if existing:
            continue

        ht_home = match.get("home_ht_score")
        ht_away = match.get("away_ht_score")

        row = TeamMatchStats(
            source_match_id=match_id,
            sport=sport_key,
            league="",
            season="",
            matchday=match.get("matchday"),
            match_date=match_date,
            team=team,
            opponent=opponent,
            is_home=is_home,
            goals_for=gf,
            goals_against=ga,
            result=_result(gf, ga),
            ht_goals_for=ht_home if is_home else ht_away,
            ht_goals_against=ht_away if is_home else ht_home,
            source=source,
        )
        db.add(row)
        count += 1

    return count


# --------------------------------------------------------------------------
# 2. Compute rolling features from TeamMatchStats → EventStatsSnapshot
# --------------------------------------------------------------------------

def compute_team_snapshot(
    team: str,
    sport: str,
    before_date: datetime,
    is_home: bool = True,
    window: int = ROLLING_WINDOW,
) -> Dict[str, Any]:
    """Compute rolling aggregate features for a team from stored match stats.

    Only uses matches before `before_date` to prevent data leakage.
    """
    with SessionLocal() as db:
        query = (
            select(TeamMatchStats)
            .where(
                and_(
                    TeamMatchStats.team == team,
                    TeamMatchStats.match_date < before_date,
                )
            )
            .order_by(TeamMatchStats.match_date.desc())
            .limit(window)
        )
        rows = list(db.scalars(query))

    if not rows:
        return _empty_snapshot(team, sport, is_home)

    # --- Basic aggregates ---
    matches_played = len(rows)
    wins = sum(1 for r in rows if r.result == "W")
    draws = sum(1 for r in rows if r.result == "D")
    losses = sum(1 for r in rows if r.result == "L")

    goals_for = [r.goals_for or 0 for r in rows]
    goals_against = [r.goals_against or 0 for r in rows]
    goals_scored_avg = sum(goals_for) / max(1, matches_played)
    goals_conceded_avg = sum(goals_against) / max(1, matches_played)
    clean_sheets = sum(1 for ga in goals_against if ga == 0)

    # --- Attack / defense strength ---
    # attack_strength = team_goals_avg / league_avg
    # defense_strength = team_conceded_avg / league_avg (lower = better defense)
    attack_strength = goals_scored_avg / LEAGUE_AVG_GOALS if LEAGUE_AVG_GOALS > 0 else 1.0
    defense_strength = goals_conceded_avg / LEAGUE_AVG_GOALS if LEAGUE_AVG_GOALS > 0 else 1.0

    # --- Form trend slope (linear regression over points per match) ---
    points = []
    for r in reversed(rows):  # chronological order
        if r.result == "W":
            points.append(3.0)
        elif r.result == "D":
            points.append(1.0)
        else:
            points.append(0.0)
    form_trend_slope = _linear_slope(points)

    # --- O2.5 and BTTS rates ---
    over25_matches = sum(1 for gf, ga in zip(goals_for, goals_against) if gf + ga > 2)
    over25_rate = over25_matches / max(1, matches_played)

    btts_matches = sum(1 for gf, ga in zip(goals_for, goals_against) if gf >= 1 and ga >= 1)
    btts_rate = btts_matches / max(1, matches_played)

    # --- Rest days (since last match) ---
    rest_days = None
    if len(rows) >= 1 and rows[0].match_date:
        delta = before_date - rows[0].match_date
        rest_days = max(0, int(delta.total_seconds() / 86400))

    # --- Schedule congestion (matches in last 14 days) ---
    cutoff_14d = before_date - timedelta(days=14)
    matches_14d = sum(1 for r in rows if r.match_date and r.match_date >= cutoff_14d)
    schedule_congestion = matches_14d / 14.0  # matches per day in window

    # --- Home / away splits ---
    home_rows = [r for r in rows if r.is_home]
    away_rows = [r for r in rows if not r.is_home]
    home_win_rate = sum(1 for r in home_rows if r.result == "W") / max(1, len(home_rows)) if home_rows else None
    away_win_rate = sum(1 for r in away_rows if r.result == "W") / max(1, len(away_rows)) if away_rows else None
    home_goals_avg = sum((r.goals_for or 0) for r in home_rows) / max(1, len(home_rows)) if home_rows else None
    away_goals_avg = sum((r.goals_for or 0) for r in away_rows) / max(1, len(away_rows)) if away_rows else None

    return {
        "team": team,
        "sport": sport,
        "is_home": is_home,
        "matches_played": matches_played,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "goals_scored_avg": round(goals_scored_avg, 4),
        "goals_conceded_avg": round(goals_conceded_avg, 4),
        "clean_sheets": clean_sheets,
        "attack_strength": round(attack_strength, 4),
        "defense_strength": round(defense_strength, 4),
        "form_trend_slope": round(form_trend_slope, 4),
        "over25_rate": round(over25_rate, 4),
        "btts_rate": round(btts_rate, 4),
        "rest_days": rest_days,
        "schedule_congestion": round(schedule_congestion, 4),
        "home_win_rate": round(home_win_rate, 4) if home_win_rate is not None else None,
        "away_win_rate": round(away_win_rate, 4) if away_win_rate is not None else None,
        "home_goals_avg": round(home_goals_avg, 4) if home_goals_avg is not None else None,
        "away_goals_avg": round(away_goals_avg, 4) if away_goals_avg is not None else None,
    }


def _empty_snapshot(team: str, sport: str, is_home: bool) -> Dict[str, Any]:
    """Return a neutral snapshot when no historical data exists."""
    return {
        "team": team,
        "sport": sport,
        "is_home": is_home,
        "matches_played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_scored_avg": 0.0,
        "goals_conceded_avg": 0.0,
        "clean_sheets": 0,
        "attack_strength": 1.0,
        "defense_strength": 1.0,
        "form_trend_slope": 0.0,
        "over25_rate": 0.0,
        "btts_rate": 0.0,
        "rest_days": None,
        "schedule_congestion": 0.0,
        "home_win_rate": None,
        "away_win_rate": None,
        "home_goals_avg": None,
        "away_goals_avg": None,
    }


def save_event_snapshot(
    event_id: str,
    team: str,
    sport: str,
    is_home: bool,
    snapshot: Dict[str, Any],
    league_position: Optional[int] = None,
    opponent_league_position: Optional[int] = None,
) -> None:
    """Upsert an EventStatsSnapshot row for a team in an upcoming event."""
    with SessionLocal() as db:
        existing = db.execute(
            select(EventStatsSnapshot).where(
                and_(
                    EventStatsSnapshot.event_id == event_id,
                    EventStatsSnapshot.team == team,
                )
            )
        ).scalar_one_or_none()

        if existing:
            # Update existing
            for key, val in snapshot.items():
                if key not in ("team", "sport", "is_home") and hasattr(existing, key):
                    setattr(existing, key, val)
            existing.league_position = league_position
            existing.opponent_league_position = opponent_league_position
            existing.snapshot_at = datetime.now(timezone.utc)
        else:
            row = EventStatsSnapshot(
                event_id=event_id,
                sport=sport,
                team=team,
                is_home=is_home,
                matches_played=snapshot.get("matches_played", 0),
                wins=snapshot.get("wins", 0),
                draws=snapshot.get("draws", 0),
                losses=snapshot.get("losses", 0),
                goals_scored_avg=snapshot.get("goals_scored_avg", 0.0),
                goals_conceded_avg=snapshot.get("goals_conceded_avg", 0.0),
                clean_sheets=snapshot.get("clean_sheets", 0),
                attack_strength=snapshot.get("attack_strength", 1.0),
                defense_strength=snapshot.get("defense_strength", 1.0),
                form_trend_slope=snapshot.get("form_trend_slope", 0.0),
                over25_rate=snapshot.get("over25_rate", 0.0),
                btts_rate=snapshot.get("btts_rate", 0.0),
                rest_days=snapshot.get("rest_days"),
                schedule_congestion=snapshot.get("schedule_congestion", 0.0),
                home_win_rate=snapshot.get("home_win_rate"),
                away_win_rate=snapshot.get("away_win_rate"),
                home_goals_avg=snapshot.get("home_goals_avg"),
                away_goals_avg=snapshot.get("away_goals_avg"),
                league_position=league_position,
                opponent_league_position=opponent_league_position,
            )
            db.add(row)
        db.commit()


def get_event_snapshot(event_id: str, team: str) -> Optional[Dict[str, Any]]:
    """Load a pre-computed EventStatsSnapshot as a feature dict."""
    with SessionLocal() as db:
        row = db.execute(
            select(EventStatsSnapshot).where(
                and_(
                    EventStatsSnapshot.event_id == event_id,
                    EventStatsSnapshot.team == team,
                )
            )
        ).scalar_one_or_none()

    if not row:
        return None

    return {
        "attack_strength": row.attack_strength or 1.0,
        "defense_strength": row.defense_strength or 1.0,
        "form_trend_slope": row.form_trend_slope or 0.0,
        "over25_rate": row.over25_rate or 0.0,
        "btts_rate": row.btts_rate or 0.0,
        "rest_days": row.rest_days,
        "schedule_congestion": row.schedule_congestion or 0.0,
        "home_win_rate": row.home_win_rate,
        "away_win_rate": row.away_win_rate,
        "home_goals_avg": row.home_goals_avg,
        "away_goals_avg": row.away_goals_avg,
        "goals_scored_avg": row.goals_scored_avg or 0.0,
        "goals_conceded_avg": row.goals_conceded_avg or 0.0,
        "clean_sheets": row.clean_sheets or 0,
        "matches_played": row.matches_played or 0,
        "league_position": row.league_position,
        "opponent_league_position": row.opponent_league_position,
    }


# --------------------------------------------------------------------------
# 3. Full ingestion pipeline
# --------------------------------------------------------------------------

def run_full_ingestion(sport_keys: Optional[List[str]] = None) -> str:
    """Run the full ingestion pipeline: fetch data + compute snapshots.

    Returns a summary string.
    """
    if not settings.stats_ingestion_enabled:
        return "Stats ingestion disabled"

    sportsdb_count = ingest_from_sportsdb(sport_keys)
    fdata_count = ingest_from_football_data(sport_keys)

    return (
        f"Ingestion complete: {sportsdb_count} SportsDB rows, "
        f"{fdata_count} football-data.org rows"
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _linear_slope(values: List[float]) -> float:
    """Compute the slope of a simple linear regression over the values.

    Positive slope = improving form, negative = declining.
    Returns 0.0 for insufficient data.
    """
    n = len(values)
    if n < 3:
        return 0.0
    x = list(range(n))
    x_mean = sum(x) / n
    y_mean = sum(values) / n
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    if denominator == 0:
        return 0.0
    return numerator / denominator
