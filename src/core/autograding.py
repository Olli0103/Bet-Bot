from __future__ import annotations

import logging
import re

from sqlalchemy import select

from src.core.elo_ratings import EloSystem
from src.core.form_tracker import update_form
from src.core.poisson_model import PoissonSoccerModel
from src.core.settings import settings
from src.core.sport_mapping import normalize_team
from src.data.postgres import SessionLocal
from src.data.models import PlacedBet
from src.integrations.odds_fetcher import OddsFetcher

log = logging.getLogger(__name__)

_elo = EloSystem()
_poisson = PoissonSoccerModel()


def _normalize_name(name: str) -> str:
    s = str(name or "").lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _selection_token(selection: str) -> str:
    s = (selection or "").strip()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    return s


def _evaluate_match_result(row: dict) -> dict:
    """Return {"winner": str, "home": str, "away": str, "home_score": int, "away_score": int, "sport": str}."""
    if not row.get("completed"):
        return {"winner": "Open"}

    scores = row.get("scores")
    if not scores:
        # Only treat as void if the match was explicitly canceled/abandoned/
        # postponed.  An empty scores array on a completed match is a
        # temporary API sync delay — returning "Open" forces a retry on
        # the next grading cycle instead of permanently voiding the bet.
        match_status = str(row.get("status") or row.get("match_status") or "").lower()
        if match_status in ("canceled", "cancelled", "abandoned", "postponed"):
            return {"winner": "Void"}
        return {"winner": "Open"}

    home_team = row.get("home_team") or ""
    away_team = row.get("away_team") or ""
    home_score, away_score = 0, 0
    norm_home = normalize_team(home_team)
    norm_away = normalize_team(away_team)

    for score_obj in scores:
        if not isinstance(score_obj, dict):
            continue
        team_name = score_obj.get("name") or ""
        try:
            points = int(score_obj.get("score", 0))
        except (ValueError, TypeError):
            points = 0

        norm_score_team = normalize_team(team_name)
        if norm_score_team == norm_home:
            home_score = points
        elif norm_score_team == norm_away:
            away_score = points

    if home_score > away_score:
        winner = str(home_team)
    elif away_score > home_score:
        winner = str(away_team)
    else:
        winner = "Draw"

    return {
        "winner": winner,
        "home": home_team,
        "away": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "sport": str(row.get("sport_key") or ""),
    }


async def run_auto_grading() -> int:
    odds = OddsFetcher()
    settled_count = 0

    # --- Phase 1: Read open bets from DB, then CLOSE session immediately ---
    # Holding a synchronous DB connection open while making async HTTP calls
    # blocks a connection-pool slot for the entire network round-trip.  With
    # SQLAlchemy's default pool of 5, a few slow API responses can exhaust
    # the pool and deadlock the entire system.
    with SessionLocal() as db:
        open_bets_rows = db.scalars(
            select(PlacedBet).where(
                PlacedBet.status == "open",
                PlacedBet.is_training_data.is_(False),
            )
        ).all()
        if not open_bets_rows:
            return 0

        # Detach ORM objects: copy the fields we need into plain dicts so
        # we can close the session before any network I/O.
        bet_snapshots = []
        for b in open_bets_rows:
            bet_snapshots.append({
                "id": b.id,
                "event_id": str(b.event_id),
                "selection": str(b.selection),
                "stake": float(b.stake),
                "odds": float(b.odds),
            })
    # DB session is now closed — connection returned to pool.

    # --- Phase 2: Async HTTP calls (no DB connection held) ---
    score_map = {}
    sports = [s.strip() for s in settings.live_sports.split(",") if s.strip()]
    available = await odds.get_sports_async()
    available_keys = {str(x.get("key")) for x in (available or []) if x.get("key")}

    expanded = []
    for s in sports:
        if s in available_keys:
            expanded.append(s)
        else:
            expanded.extend(sorted([k for k in available_keys if k.startswith(s + "_")]))

    for sport in expanded:
        try:
            rows = await odds.get(
                f"sports/{sport}/scores",
                params={"apiKey": settings.odds_api_key, "daysFrom": 3},
            )
        except Exception as exc:
            log.warning(
                "Scores fetch failed for sport '%s': %s", sport, exc,
            )
            continue
        if not isinstance(rows, list):
            log.warning(
                "Unexpected scores response type for '%s': %s",
                sport, type(rows).__name__,
            )
            continue

        for r in rows:
            r["sport_key"] = sport
            event_id = str(r.get("id") or "")
            result = _evaluate_match_result(r)
            if event_id and result.get("winner") != "Open":
                score_map[event_id] = result

    # --- Phase 3: Match bets to results (in-memory) ---
    updates = []  # list of (bet_id, status, pnl, result_dict, pick)
    updated_events: set = set()

    for snap in bet_snapshots:
        result = score_map.get(snap["event_id"])
        if not result:
            continue

        winner = result["winner"]

        # Void only for explicitly canceled/abandoned/postponed matches.
        # An empty scores array with completed=True is a temporary API
        # sync delay, NOT a void — grading must be retried later.
        if winner == "Void":
            updates.append((snap["id"], "void", 0.0, result, None))
            continue

        pick = _selection_token(snap["selection"])
        is_won = normalize_team(pick) == normalize_team(winner)
        if is_won:
            status = "won"
            pnl = round(snap["stake"] * (snap["odds"] - 1.0), 2)
        else:
            status = "lost"
            pnl = round(-snap["stake"], 2)

        updates.append((snap["id"], status, pnl, result, pick))

    if not updates:
        return 0

    # --- Phase 4: Persist results in a NEW, short-lived DB session ---
    with SessionLocal() as db:
        for bet_id, status, pnl, result, pick in updates:
            bet = db.get(PlacedBet, bet_id)
            if bet is None or bet.status != "open":
                continue

            bet.status = status
            bet.pnl = pnl
            settled_count += 1

            if pick is not None:
                is_won = status == "won"
                try:
                    update_form(pick, is_won)
                except Exception as exc:
                    log.warning("Form tracker update failed for %s: %s", pick, exc)

            # Update Elo and Poisson (once per event)
            ev_id = str(bet.event_id)
            winner = result["winner"]
            if ev_id not in updated_events:
                updated_events.add(ev_id)
                home = result.get("home", "")
                away = result.get("away", "")
                sport_key = result.get("sport", "")
                home_won = winner == home

                if home and away:
                    try:
                        _elo.update(home, away, home_won, sport=sport_key)
                    except Exception as exc:
                        log.warning("Elo update failed for %s vs %s: %s", home, away, exc)

                    # Poisson update for soccer (needs actual scores)
                    if sport_key.startswith("soccer"):
                        try:
                            _poisson.update_strengths(
                                home, away,
                                result.get("home_score", 0),
                                result.get("away_score", 0),
                            )
                        except Exception as exc:
                            log.warning("Poisson update failed for %s vs %s: %s", home, away, exc)

        db.commit()

    return settled_count
