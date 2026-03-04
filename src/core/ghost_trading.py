import logging
from datetime import datetime, timezone
from typing import Dict, Optional

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


def auto_place_virtual_bets(signals: list, features_dict: dict):
    """Auto-place virtual bets for positive EV signals.

    Parameters
    ----------
    signals : list of BetSignal
    features_dict : dict keyed by ``"{event_id}:{selection}"``
    """
    placed_count = 0
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    with SessionLocal() as db:
        existing = db.query(PlacedBet.event_id, PlacedBet.selection).filter(PlacedBet.created_at >= start_of_day).all()
        existing_set = {f"{e[0]}|{e[1]}" for e in existing}

        for sig in signals:
            sig_key = f"{sig.event_id}|{sig.selection}"
            if sig.expected_value > 0 and sig_key not in existing_set:
                feat = features_dict.get(f"{sig.event_id}:{sig.selection}", {})
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
                    # Persist ALL ML features so the trainer can use them
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
) -> bool:
    """Place a single virtual (ghost) bet.

    Called from the Executioner agent and the Telegram "Ghost Bet" button.
    """
    features = features or {}
    now = datetime.now(timezone.utc)

    try:
        with SessionLocal() as db:
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
                # Persist ALL ML features so the trainer can use them
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
