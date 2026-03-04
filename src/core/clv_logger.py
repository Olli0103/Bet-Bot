"""Closing Line Value (CLV) Logger.

Fetches Pinnacle sharp closing odds exactly at kickoff and persists them
to the ``event_closing_lines`` table.  Also back-fills
``PlacedBet.sharp_closing_odds / sharp_closing_prob`` for bets on those
events so the ML trainer can use CLV as a regression target.

Called by the JIT scheduler in ``core_worker.py`` 1 minute before kickoff.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.feature_engineering import FeatureEngineer
from src.core.settings import settings
from src.data.models import EventClosingLine, PlacedBet
from src.data.postgres import SessionLocal
from src.data.redis_cache import cache
from src.integrations.odds_fetcher import OddsFetcher

log = logging.getLogger(__name__)

SHARP_PRIORITY = ["pinnacle", "betfair_ex_uk", "bet365"]


def _pick_sharp(
    bookmakers: List[Dict[str, Any]],
    market_key: str = "h2h",
) -> Tuple[str, Dict[str, float], float]:
    """Extract sharp book prices + vig from raw bookmaker data.

    Returns (sharp_book_name, {selection: odds}, overround).
    """
    for bk_name in SHARP_PRIORITY:
        for bm in bookmakers:
            if (bm.get("key") or "").lower() != bk_name:
                continue
            for mkt in bm.get("markets") or []:
                if (mkt.get("key") or "") != market_key:
                    continue
                prices: Dict[str, float] = {}
                for o in mkt.get("outcomes") or []:
                    name = o.get("name")
                    price = o.get("price")
                    if name and isinstance(price, (int, float)) and price > 1.0:
                        prices[name] = float(price)
                if prices:
                    vig = FeatureEngineer.calculate_vig(prices)
                    return bk_name, prices, vig
    return "", {}, 0.0


def log_closing_lines(
    sport_key: str,
    events: List[Dict[str, Any]],
    model_predictions: Optional[Dict[str, Dict[str, float]]] = None,
) -> int:
    """Persist closing lines for a list of raw Odds-API events.

    Parameters
    ----------
    sport_key : str
        The sport key (e.g. ``soccer_epl``).
    events : list
        Raw events from the Odds API ``/v4/sports/{key}/odds`` endpoint.
    model_predictions : dict, optional
        ``{event_id: {selection: model_prob}}`` from cached signals.

    Returns
    -------
    int
        Number of closing lines logged.
    """
    model_predictions = model_predictions or {}
    rows_logged = 0

    with SessionLocal() as db:
        try:
            for event in events:
                event_id = str(event.get("id") or "")
                home = event.get("home_team") or ""
                away = event.get("away_team") or ""
                commence = event.get("commence_time") or ""
                if not event_id:
                    continue

                commence_dt = None
                try:
                    commence_dt = datetime.fromisoformat(
                        str(commence).replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

                sharp_book, sharp_prices, vig = _pick_sharp(
                    event.get("bookmakers") or []
                )
                if not sharp_book or not sharp_prices:
                    continue

                overround = 1.0 + vig
                event_preds = model_predictions.get(event_id, {})

                for selection, odds in sharp_prices.items():
                    raw_prob = 1.0 / odds
                    closing_prob = raw_prob / overround if overround > 0 else raw_prob

                    model_prob = event_preds.get(selection)
                    model_ev = event_preds.get(f"{selection}_ev")

                    stmt = pg_insert(EventClosingLine).values(
                        event_id=event_id,
                        sport=sport_key,
                        selection=selection,
                        market=market_key,
                        home_team=home,
                        away_team=away,
                        sharp_book=sharp_book,
                        closing_odds=odds,
                        closing_implied_prob=round(closing_prob, 6),
                        closing_vig=round(vig, 4),
                        model_prob_at_signal=round(model_prob, 4) if model_prob else None,
                        model_ev_at_signal=round(model_ev, 4) if model_ev else None,
                        commence_time=commence_dt,
                    ).on_conflict_do_update(
                        constraint="uq_closing_event_sel_mkt",
                        set_={
                            "closing_odds": odds,
                            "closing_implied_prob": round(closing_prob, 6),
                            "closing_vig": round(vig, 4),
                            "logged_at": datetime.now(timezone.utc),
                        },
                    )
                    db.execute(stmt)
                    rows_logged += 1

                    # Back-fill PlacedBet rows for this event+selection
                    db.execute(
                        update(PlacedBet)
                        .where(
                            PlacedBet.event_id == event_id,
                            PlacedBet.sharp_closing_odds.is_(None),
                        )
                        .values(
                            sharp_closing_odds=odds,
                            sharp_closing_prob=round(closing_prob, 6),
                        )
                    )

            db.commit()
        except Exception:
            db.rollback()
            log.error("Failed to log closing lines", exc_info=True)
            raise

    log.info("Logged %d closing lines for %s", rows_logged, sport_key)
    return rows_logged


def fetch_and_log_closing_lines(sport_key: str) -> int:
    """Fetch current sharp odds for a sport and log them as closing lines.

    This is the JIT entry-point called ~1 minute before kickoff.
    """
    odds = OddsFetcher()
    try:
        events = odds.get_sport_odds(
            sport_key=sport_key,
            regions="eu",
            markets="h2h",
            ttl_seconds=30,  # very short cache — we want fresh data
        )
    except Exception:
        log.error("Failed to fetch closing odds for %s", sport_key, exc_info=True)
        return 0

    if not isinstance(events, list):
        return 0

    # Load cached model predictions so we can record model_prob at signal time
    model_predictions: Dict[str, Dict[str, float]] = {}
    try:
        from src.core.live_feed import get_all_ranked_signals
        items, _ = get_all_ranked_signals()
        for item in items:
            eid = item.get("event_id", "")
            sel_raw = item.get("selection", "")
            # Strip " (Home vs Away)" suffix if present
            sel = sel_raw.split(" (")[0] if " (" in sel_raw else sel_raw
            if eid:
                if eid not in model_predictions:
                    model_predictions[eid] = {}
                model_predictions[eid][sel] = item.get("model_probability", 0)
                model_predictions[eid][f"{sel}_ev"] = item.get("expected_value", 0)
    except Exception:
        pass

    return log_closing_lines(sport_key, events, model_predictions)
