import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.data.postgres import SessionLocal
from src.data.models import PlacedBet

log = logging.getLogger(__name__)


def _safe_meta(feat: dict) -> dict:
    """Return a JSON-safe copy of the feature dict for JSONB storage."""
    if not feat:
        return {}
    return {k: float(v) if isinstance(v, (int, float)) else v
            for k, v in feat.items()}


def _user_bet_sources() -> tuple:
    """Data source values that count as user-placed bets for duplicate checks.

    paper_signal and historical_import are excluded — they should never
    block a user from placing a real bet on the same event.
    """
    return ("live_trade", "manual")


def auto_place_virtual_bets(
    signals: list,
    features_dict: dict,
    owner_chat_id: str = "",
):
    """Auto-place virtual bets for positive EV signals.

    Parameters
    ----------
    signals : list of BetSignal
    features_dict : dict keyed by ``"{event_id}:{selection}"``
    owner_chat_id : str
        Owner's chat ID for portfolio isolation.
    """
    placed_count = 0
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    with SessionLocal() as db:
        # Owner-scoped duplicate check: only consider user bets (not paper/import)
        query = select(PlacedBet.event_id, PlacedBet.selection).where(
            PlacedBet.created_at >= start_of_day,
            PlacedBet.data_source.in_(_user_bet_sources()),
        )
        if owner_chat_id:
            query = query.where(PlacedBet.owner_chat_id == owner_chat_id)
        existing = db.execute(query).all()
        existing_set = {f"{e[0]}|{e[1]}" for e in existing}

        for sig in signals:
            sig_key = f"{sig.event_id}|{sig.selection}"
            if sig.expected_value <= 0:
                continue

            feat = features_dict.get(f"{sig.event_id}:{sig.selection}", {})

            if sig_key in existing_set:
                # Refresh enrichment/features for already-open live trades
                existing_bet = db.scalar(
                    select(PlacedBet).where(
                        PlacedBet.event_id == str(sig.event_id),
                        PlacedBet.selection == str(sig.selection),
                        PlacedBet.market == str(sig.market),
                        PlacedBet.data_source.in_(_user_bet_sources()),
                        *([PlacedBet.owner_chat_id == owner_chat_id] if owner_chat_id else []),
                    )
                )
                if existing_bet:
                    existing_bet.meta_features = {**(existing_bet.meta_features or {}), **_safe_meta(feat)}
                    existing_bet.sharp_implied_prob = float(feat.get("sharp_implied_prob", existing_bet.sharp_implied_prob or 0.0))
                    existing_bet.sharp_vig = float(feat.get("sharp_vig", existing_bet.sharp_vig or 0.0))
                    existing_bet.sentiment_delta = float(feat.get("sentiment_delta", existing_bet.sentiment_delta or 0.0))
                    existing_bet.injury_delta = float(feat.get("injury_delta", existing_bet.injury_delta or 0.0))
                    existing_bet.form_winrate_l5 = float(feat.get("form_winrate_l5", existing_bet.form_winrate_l5 or 0.5))
                    existing_bet.form_games_l5 = int(feat.get("form_games_l5", existing_bet.form_games_l5 or 0))
                continue

            new_bet = PlacedBet(
                event_id=str(sig.event_id),
                sport=str(sig.sport),
                market=str(sig.market),
                selection=str(sig.selection),
                odds=float(sig.bookmaker_odds),
                odds_open=float(sig.bookmaker_odds),
                odds_close=float(sig.bookmaker_odds),
                clv=float(feat.get("clv", 0.0)),
                form_winrate_l5=float(feat.get("form_winrate_l5", 0.5)),
                form_games_l5=int(feat.get("form_games_l5", 0)),
                stake=float(sig.recommended_stake),
                status="open",
                sharp_implied_prob=float(feat.get("sharp_implied_prob", 0.0)),
                sharp_vig=float(feat.get("sharp_vig", 0.0)),
                sentiment_delta=float(feat.get("sentiment_delta", 0.0)),
                injury_delta=float(feat.get("injury_delta", 0.0)),
                is_training_data=False,
                data_source="live_trade",
                owner_chat_id=owner_chat_id or None,
                meta_features=_safe_meta(feat),
            )
            db.add(new_bet)
            placed_count += 1
        db.commit()

    return placed_count


def place_virtual_bet(
    event_id: str,
    sport: str,
    market: str = "h2h",
    selection: str = "",
    odds: float = 2.0,
    stake: float = 1.0,
    features: Optional[Dict] = None,
    owner_chat_id: str = "",
) -> bool:
    """Place a single virtual (ghost) bet.

    Called from the paper-trading pipeline for tracking purposes.
    Duplicate check is scoped to owner + user bet sources only.
    """
    features = features or {}

    try:
        with SessionLocal() as db:
            # Owner-scoped duplicate check (only user bets, not paper/import)
            existing = db.scalar(
                select(PlacedBet.id).where(
                    PlacedBet.event_id == event_id,
                    PlacedBet.selection == selection,
                    PlacedBet.market == market,
                    PlacedBet.data_source.in_(_user_bet_sources()),
                    *([PlacedBet.owner_chat_id == owner_chat_id] if owner_chat_id else []),
                )
            )
            if existing:
                # Update enrichment/features on duplicate instead of skipping silently
                existing_bet = db.get(PlacedBet, existing)
                if existing_bet:
                    safe_feat = _safe_meta(features)
                    existing_bet.meta_features = {**(existing_bet.meta_features or {}), **safe_feat}
                    existing_bet.sharp_implied_prob = float(features.get("sharp_implied_prob", existing_bet.sharp_implied_prob or 0.0))
                    existing_bet.sharp_vig = float(features.get("sharp_vig", existing_bet.sharp_vig or 0.0))
                    existing_bet.sentiment_delta = float(features.get("sentiment_delta", existing_bet.sentiment_delta or 0.0))
                    existing_bet.injury_delta = float(features.get("injury_delta", existing_bet.injury_delta or 0.0))
                    existing_bet.form_winrate_l5 = float(features.get("form_winrate_l5", existing_bet.form_winrate_l5 or 0.5))
                    existing_bet.form_games_l5 = int(features.get("form_games_l5", existing_bet.form_games_l5 or 0))
                    db.commit()
                log.info("Duplicate bet updated (owner-scoped): event=%s sel=%s owner=%s",
                         event_id, selection, owner_chat_id or "global")
                return False

            new_bet = PlacedBet(
                event_id=str(event_id),
                sport=str(sport),
                market=str(market),
                selection=str(selection),
                odds=float(odds),
                odds_open=float(odds),
                odds_close=float(odds),
                clv=float(features.get("clv", 0.0)),
                form_winrate_l5=float(features.get("form_winrate_l5", 0.5)),
                form_games_l5=int(features.get("form_games_l5", 0)),
                stake=float(stake),
                status="open",
                sharp_implied_prob=float(features.get("sharp_implied_prob", 0.0)),
                sharp_vig=float(features.get("sharp_vig", 0.0)),
                sentiment_delta=float(features.get("sentiment_delta", 0.0)),
                injury_delta=float(features.get("injury_delta", 0.0)),
                is_training_data=False,
                data_source="live_trade",
                owner_chat_id=owner_chat_id or None,
                meta_features=_safe_meta(features),
            )
            db.add(new_bet)
            db.commit()
        return True
    except IntegrityError:
        log.info("Duplicate bet skipped: event=%s sel=%s market=%s", event_id, selection, market)
        return False
    except Exception as exc:
        log.warning("place_virtual_bet failed: %s", exc)
        return False
