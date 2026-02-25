from datetime import datetime, timezone

from src.data.postgres import SessionLocal
from src.data.models import PlacedBet


def auto_place_virtual_bets(signals: list, features_dict: dict):
    placed_count = 0
    now = datetime.now(timezone.utc)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

    with SessionLocal() as db:
        existing = db.query(PlacedBet.event_id, PlacedBet.selection).filter(PlacedBet.created_at >= start_of_day).all()
        existing_set = {f"{e[0]}|{e[1]}" for e in existing}

        for sig in signals:
            sig_key = f"{sig.event_id}|{sig.selection}"
            if sig.expected_value > 0 and sig_key not in existing_set:
                feat = features_dict.get(sig.event_id, {})
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
                    form_games_l5=int(feat.get("form_games_l5", 0.0)),
                    stake=float(sig.recommended_stake),
                    status="open",
                    sharp_implied_prob=float(feat.get("sharp_prob", 0.0)),
                    sharp_vig=float(feat.get("sharp_vig", 0.0)),
                    sentiment_delta=float(feat.get("sentiment", 0.0)),
                    injury_delta=float(feat.get("injuries", 0.0)),
                )
                db.add(new_bet)
                placed_count += 1
        db.commit()

    return placed_count
