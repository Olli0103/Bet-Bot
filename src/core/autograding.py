from __future__ import annotations

import logging
import re

from sqlalchemy import select

from src.core.form_tracker import update_form
from src.core.settings import settings
from src.data.postgres import SessionLocal
from src.data.models import PlacedBet
from src.integrations.odds_fetcher import OddsFetcher

log = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    s = str(name or "").lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _selection_token(selection: str) -> str:
    s = (selection or "").strip()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    return s


def _evaluate_match_result(row: dict) -> str:
    if not row.get("completed"):
        return "Open"

    scores = row.get("scores")
    if not scores:
        return "Void"

    home_team = row.get("home_team") or ""
    away_team = row.get("away_team") or ""
    home_score, away_score = 0, 0

    for score_obj in scores:
        if not isinstance(score_obj, dict):
            continue
        team_name = score_obj.get("name")
        try:
            points = int(score_obj.get("score", 0))
        except (ValueError, TypeError):
            points = 0

        if team_name == home_team:
            home_score = points
        elif team_name == away_team:
            away_score = points

    if home_score > away_score:
        return str(home_team)
    if away_score > home_score:
        return str(away_team)
    return "Draw"


async def run_auto_grading() -> int:
    odds = OddsFetcher()
    settled_count = 0

    with SessionLocal() as db:
        open_bets = db.scalars(select(PlacedBet).where(PlacedBet.status == "open")).all()
        if not open_bets:
            return 0

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
            except Exception:
                continue
            if not isinstance(rows, list):
                continue

            for r in rows:
                event_id = str(r.get("id") or "")
                winner = _evaluate_match_result(r)
                if event_id and winner != "Open":
                    score_map[event_id] = winner

        for bet in open_bets:
            winner = score_map.get(str(bet.event_id))
            if not winner:
                continue

            if winner == "Void":
                bet.status = "void"
                bet.pnl = 0.0
                settled_count += 1
                continue

            pick = _selection_token(str(bet.selection))
            is_won = _normalize_name(pick) == _normalize_name(winner)
            if is_won:
                bet.status = "won"
                bet.pnl = round(float(bet.stake) * (float(bet.odds) - 1.0), 2)
            else:
                bet.status = "lost"
                bet.pnl = round(-float(bet.stake), 2)
            settled_count += 1

            # Update form tracker for the selected team
            try:
                update_form(pick, is_won)
            except Exception as exc:
                log.warning("Form tracker update failed for %s: %s", pick, exc)

        db.commit()

    return settled_count
