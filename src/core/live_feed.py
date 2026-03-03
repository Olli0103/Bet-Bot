from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.core.bankroll import BankrollManager
from src.core.betting_engine import BettingEngine
from src.core.betting_math import public_bias_score
from src.core.elo_ratings import EloSystem
from src.core.enrichment import batch_team_sentiment, soccer_injury_delta
from src.core.feature_engineering import FeatureEngineer
from src.core.form_tracker import get_form_l5
from src.core.ghost_trading import auto_place_virtual_bets
from src.core.h2h_tracker import get_h2h_features
from src.core.poisson_model import PoissonSoccerModel
from src.core.pricing_model import QuantPricingModel
from src.core.settings import settings
from src.core.volatility_tracker import (
    detect_steam_move,
    get_volatility,
    get_volatility_features,
    record_odds_snapshot,
)
from src.data.redis_cache import cache
from src.integrations.odds_fetcher import OddsFetcher
from src.models.betting import BetSignal, ComboBet

log = logging.getLogger(__name__)

SNAPSHOT_KEY = "live_snapshot:value_bets"
COMBO_KEY = "live_snapshot:combo_bets"
COMBO_LEGS_KEY = "live_snapshot:combo_legs"
META_KEY = "live_snapshot:meta"

SHARP_PRIORITY = ["pinnacle", "betfair_ex_uk", "bet365"]
SHARP_WEIGHTS = {"pinnacle": 0.50, "betfair_ex_uk": 0.30, "bet365": 0.20}


def _bet_window(now_utc: datetime) -> Tuple[datetime, datetime]:
    tz = ZoneInfo("Europe/Berlin")
    local = now_utc.astimezone(tz)
    start = local.replace(hour=7, minute=0, second=0, microsecond=0)
    if local < start:
        start = start - timedelta(days=1)
    end = start + timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _extract_prices(
    bookmakers: List[Dict[str, Any]], market_key: str = "h2h"
) -> Dict[str, Dict[str, float]]:
    """Extract prices per bookmaker for a given market key (h2h, spreads, totals)."""
    out: Dict[str, Dict[str, float]] = {}
    for b in bookmakers or []:
        bk = (b.get("key") or "").lower()
        for m in b.get("markets", []) or []:
            if (m.get("key") or "") != market_key:
                continue
            prices: Dict[str, float] = {}
            for o in m.get("outcomes", []) or []:
                name = o.get("name")
                price = o.get("price")
                point = o.get("point")
                if name and isinstance(price, (int, float)):
                    label = f"{name}|{point}" if point is not None else name
                    prices[label] = float(price)
            if prices:
                out[bk] = prices
    return out


def _consensus_sharp_prob(
    all_book_prices: Dict[str, Dict[str, float]], selection: str
) -> Optional[float]:
    """Weighted consensus implied probability from sharp books."""
    total_weight = 0.0
    weighted_prob = 0.0
    for book, weight in SHARP_WEIGHTS.items():
        prices = all_book_prices.get(book, {})
        price = prices.get(selection)
        if price and price > 1.0:
            weighted_prob += weight * (1.0 / price)
            total_weight += weight
    if total_weight <= 0:
        return None
    return round(weighted_prob / total_weight, 4)


def _get_weather_features(home: str, away: str, sport: str) -> Dict[str, float]:
    """Fetch weather features for outdoor sports. Returns empty dict on failure."""
    if not sport.startswith(("soccer", "tennis")):
        return {"weather_rain": 0.0, "weather_wind_high": 0.0}
    try:
        from src.data.venue_coordinates import get_venue_coordinates as get_coordinates
        from src.integrations.weather_fetcher import OpenMeteoFetcher

        coords = get_coordinates(home)
        if not coords:
            return {"weather_rain": 0.0, "weather_wind_high": 0.0}
        lat, lng = coords
        fetcher = OpenMeteoFetcher()
        data = fetcher.get_hourly_weather(lat, lng)
        hourly = data.get("hourly", {})
        precip = hourly.get("precipitation", [0])
        wind = hourly.get("wind_speed_10m", [0])
        rain = 1.0 if any(p > 1.0 for p in precip[:6]) else 0.0
        wind_high = 1.0 if any(w > 40 for w in wind[:6]) else 0.0
        return {"weather_rain": rain, "weather_wind_high": wind_high}
    except Exception:
        return {"weather_rain": 0.0, "weather_wind_high": 0.0}


def _extract_last_updates(bookmakers: List[Dict[str, Any]]) -> Dict[str, str]:
    """Extract last_update timestamps per bookmaker for staleness detection."""
    out: Dict[str, str] = {}
    for b in bookmakers or []:
        bk = (b.get("key") or "").lower()
        ts = b.get("last_update") or ""
        if bk and ts:
            out[bk] = ts
    return out


def _compute_staleness_minutes(last_updates: Dict[str, str], target_book: str, sharp_book: str) -> float:
    """Compute how many minutes behind the target book is relative to the sharp book."""
    target_ts = last_updates.get(target_book, "")
    sharp_ts = last_updates.get(sharp_book, "")
    if not target_ts or not sharp_ts:
        return 0.0
    try:
        t_target = datetime.fromisoformat(target_ts.replace("Z", "+00:00"))
        t_sharp = datetime.fromisoformat(sharp_ts.replace("Z", "+00:00"))
        delta = (t_sharp - t_target).total_seconds() / 60.0
        return max(0.0, delta)
    except Exception:
        return 0.0


def _pick_sharp(prices: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    for b in SHARP_PRIORITY:
        if prices.get(b):
            return b, prices[b]
    return "", {}


def _pick_target_book(prices: Dict[str, Dict[str, float]], sharp_book: str) -> Tuple[str, Dict[str, float]]:
    if prices.get("tipico_de"):
        return "tipico_de", prices["tipico_de"]

    best_book = ""
    best_avg = -1.0
    for bk, outcomes in prices.items():
        if bk == sharp_book:
            continue
        if not outcomes:
            continue
        avg = sum(outcomes.values()) / max(1, len(outcomes))
        if avg > best_avg:
            best_avg = avg
            best_book = bk
    return (best_book, prices.get(best_book, {})) if best_book else ("", {})


def _build_top_combos(engine: BettingEngine, legs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build optimized combos using ComboOptimizer with dynamic correlation."""
    from src.core.combo_optimizer import ComboOptimizer
    optimizer = ComboOptimizer(engine)
    return optimizer.build_all_combos(legs)


def fetch_and_build_signals(bankroll: float | None = None) -> List[BetSignal]:
    from src.core.dynamic_settings import dynamic_settings

    # Use dynamic settings for active sports, falling back to env var
    dyn_sports = dynamic_settings.get_active_sports()
    sports = dyn_sports if dyn_sports else [s.strip() for s in settings.live_sports.split(",") if s.strip()]
    active_markets = set(dynamic_settings.get_active_markets())
    min_odds_threshold = dynamic_settings.get_min_odds()
    combo_sizes = dynamic_settings.get_combo_sizes()

    odds = OddsFetcher()

    # Dynamic bankroll: use BankrollManager if no explicit bankroll passed
    if bankroll is None:
        try:
            bankroll = BankrollManager().get_current_bankroll()
        except Exception:
            bankroll = settings.initial_bankroll

    engine = BettingEngine(bankroll=bankroll)

    signals: List[BetSignal] = []
    combo_legs: List[Dict[str, Any]] = []
    features: Dict[str, Dict[str, float]] = {}
    qpm = QuantPricingModel()
    elo = EloSystem()
    poisson = PoissonSoccerModel()

    # Determine tax rate based on target book
    tax_rate = 0.0 if settings.tax_free_mode else settings.tipico_tax_rate

    # Resolve user base-keys (e.g. "tennis_atp") into exact in-season API keys
    # (e.g. "tennis_atp_dubai", "tennis_atp_wimbledon") via the /v4/sports endpoint.
    try:
        api_active = odds.get_active_sports_from_api()
    except Exception:
        api_active = []

    expanded_sports = OddsFetcher.resolve_sport_keys(sports, api_active)

    stats = {"sports_requested": len(sports), "sports_expanded": len(expanded_sports), "events_seen": 0, "signals": 0}

    raw_events: List[Tuple[str, Dict[str, Any]]] = []
    # Fetch historical odds for momentum calculation (Pro API)
    momentum_data: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sport in expanded_sports:
        try:
            events = odds.get_sport_odds(sport_key=sport, regions="eu", markets="h2h,spreads,totals", ttl_seconds=120)
        except Exception:
            continue
        if not isinstance(events, list):
            continue

        # Fetch 12h-ago odds for momentum (best-effort, non-blocking)
        try:
            momentum_data[sport] = odds.get_odds_12h_ago(sport_key=sport)
        except Exception:
            momentum_data[sport] = {}

        for e in events:
            stats["events_seen"] += 1
            raw_events.append((sport, e))

    # --- Enrichment: sentiment analysis ---
    sentiment: Dict[str, float] = {}
    injury_sentiment: Dict[str, float] = {}
    if settings.enrichment_enabled:
        all_teams: List[str] = []
        for _, e in raw_events:
            home = e.get("home_team") or ""
            away = e.get("away_team") or ""
            if home:
                all_teams.append(home)
            if away:
                all_teams.append(away)
        try:
            sentiment = batch_team_sentiment([t for t in all_teams if t], max_teams=24)
        except Exception as exc:
            log.warning("Batch sentiment enrichment failed: %s", exc)

        # --- Injury news enrichment (Rotowire RSS, best-effort) ---
        try:
            from src.integrations.rss_fetcher import RSSFetcher
            rss = RSSFetcher()
            unique_teams = list({t for t in all_teams if t})
            for sport_key in expanded_sports:
                try:
                    rss_hits = rss.fetch_injury_news(sport_key, unique_teams[:20], max_age_hours=24)
                    for hit in rss_hits:
                        matched = hit.get("team_match", "")
                        if matched:
                            injury_sentiment[matched] = -0.8  # injury news = strong negative
                except Exception:
                    pass
        except Exception as exc:
            log.warning("RSS injury enrichment failed: %s", exc)

    now_utc = datetime.now(timezone.utc)
    _, window_end = _bet_window(now_utc)

    for sport, e in raw_events:
        event_id = str(e.get("id") or "")
        home = e.get("home_team") or ""
        away = e.get("away_team") or ""
        commence = str(e.get("commence_time") or "")

        try:
            ct = datetime.fromisoformat(commence.replace("Z", "+00:00")).astimezone(timezone.utc)
            if ct < now_utc or ct >= window_end:
                continue
        except Exception:
            continue

        # Time to kickoff in hours
        time_to_kickoff = max(0.0, (ct - now_utc).total_seconds() / 3600.0)

        # --- H2H prices (primary market) ---
        prices = _extract_prices(e.get("bookmakers") or [])
        last_updates = _extract_last_updates(e.get("bookmakers") or [])
        sharp_book, sharp = _pick_sharp(prices)
        if not sharp_book or not sharp:
            continue
        target_book, target = _pick_target_book(prices, sharp_book)
        if not target_book or not target:
            continue

        source_mode = "primary" if target_book == "tipico_de" and sharp_book == "pinnacle" else (
            "fallback_tipico_sharp_proxy" if target_book == "tipico_de" else "fallback_best_vs_sharp"
        )
        conf = 1.0 if source_mode == "primary" else 0.8 if source_mode == "fallback_tipico_sharp_proxy" else 0.65

        # --- Line staleness: boost confidence if Tipico is stale vs sharp ---
        staleness_min = _compute_staleness_minutes(last_updates, target_book, sharp_book)
        if staleness_min > 15:
            conf = min(1.0, conf + 0.15)

        # --- Injury enrichment for soccer ---
        inj_home = inj_away = 0
        if settings.enrichment_enabled and sport.startswith("soccer"):
            try:
                inj_home, inj_away = soccer_injury_delta(home, away, commence)
            except Exception:
                inj_home = inj_away = 0

        # --- Form tracking ---
        try:
            home_wr, home_gp = get_form_l5(home)
            away_wr, away_gp = get_form_l5(away)
        except Exception:
            home_wr, home_gp = 0.5, 0
            away_wr, away_gp = 0.5, 0

        # --- Elo ratings ---
        try:
            elo_feats = elo.get_elo_features(home, away)
        except Exception:
            elo_feats = {"elo_diff": 0.0, "elo_expected": 0.5}

        # --- H2H history ---
        try:
            h2h_feats = get_h2h_features(home, away)
        except Exception:
            h2h_feats = {"h2h_home_winrate": 0.5}

        # --- Volatility tracking ---
        try:
            vol_feats = get_volatility_features(home, away)
        except Exception:
            vol_feats = {"home_volatility": 0.0, "away_volatility": 0.0}

        # --- Weather (outdoor sports only) ---
        weather_feats = _get_weather_features(home, away, sport)

        # --- Poisson model for soccer ---
        poisson_pred: Optional[Dict[str, float]] = None
        if sport.startswith("soccer"):
            try:
                poisson_pred = poisson.predict_match(home, away)
            except Exception:
                poisson_pred = None

        # --- Injury sentiment delta (from Rotowire RSS enrichment) ---
        inj_sent_home = injury_sentiment.get(home, 0.0)
        inj_sent_away = injury_sentiment.get(away, 0.0)

        # Determine effective tax rate (only for Tipico)
        effective_tax = tax_rate if target_book == "tipico_de" else 0.0

        # Compute public bias per selection (Tipico vs sharp shading)
        bias_scores = public_bias_score(sharp, target) if target_book == "tipico_de" else {}

        # Momentum data for this sport
        sport_momentum = momentum_data.get(sport, {})

        for selection, target_odds in target.items():
            sharp_odds = sharp.get(selection)
            if not sharp_odds:
                continue

            # Apply min odds threshold from dynamic settings
            if float(target_odds) < min_odds_threshold:
                continue

            # Skip markets not in active_markets
            if "h2h" not in active_markets:
                continue

            is_home = selection == home
            sel_wr = home_wr if is_home else away_wr
            sel_gp = home_gp if is_home else away_gp

            # Record odds snapshot for volatility tracking
            try:
                record_odds_snapshot(selection, 1.0 / target_odds if target_odds > 1.0 else 0.0)
            except Exception:
                pass

            # Steam move detection
            steam = False
            try:
                team_vol = get_volatility(selection)
                steam = detect_steam_move(target_odds, sharp_odds, team_vol)
            except Exception:
                pass

            # Poisson probability for this selection (soccer only)
            poisson_prob = None
            if poisson_pred and sport.startswith("soccer"):
                if is_home:
                    poisson_prob = poisson_pred.get("h2h_home")
                elif selection == "Draw":
                    poisson_prob = poisson_pred.get("h2h_draw")
                else:
                    poisson_prob = poisson_pred.get("h2h_away")

            inj_delta_sent = (inj_sent_home - inj_sent_away) if is_home else (inj_sent_away - inj_sent_home)

            # Market momentum: implied probability delta over 12h (Pro API)
            current_ip = 1.0 / float(sharp_odds) if float(sharp_odds) > 1.0 else 0.5
            historical_ip = sport_momentum.get(event_id, {}).get(selection, current_ip)
            market_momentum = current_ip - historical_ip

            ml_features = FeatureEngineer.build_core_features(
                target_odds=float(target_odds),
                sharp_odds=float(sharp_odds),
                sharp_market=sharp,
                sentiment_home=float(sentiment.get(home, 0.0)),
                sentiment_away=float(sentiment.get(away, 0.0)),
                injuries_home=int(inj_home),
                injuries_away=int(inj_away),
                selection=selection,
                home_team=home,
                form_winrate_l5=sel_wr,
                form_games_l5=sel_gp,
                elo_diff=elo_feats.get("elo_diff", 0.0),
                elo_expected=elo_feats.get("elo_expected", 0.5),
                h2h_home_winrate=h2h_feats.get("h2h_home_winrate", 0.5),
                weather_rain=weather_feats.get("weather_rain", 0.0),
                weather_wind_high=weather_feats.get("weather_wind_high", 0.0),
                home_volatility=vol_feats.get("home_volatility", 0.0),
                away_volatility=vol_feats.get("away_volatility", 0.0),
                is_steam_move=steam,
                line_staleness=staleness_min,
                poisson_true_prob=poisson_prob,
                injury_news_delta=inj_delta_sent,
                time_to_kickoff_hours=time_to_kickoff,
                public_bias=bias_scores.get(selection, 0.0),
                market_momentum=market_momentum,
            )

            model_p = qpm.get_true_probability(
                sharp_prob=ml_features["sharp_implied_prob"],
                sentiment=ml_features["sentiment_delta"],
                injuries=ml_features["injury_delta"],
                clv=ml_features["clv"],
                sharp_vig=ml_features["sharp_vig"],
                form_winrate_l5=ml_features["form_winrate_l5"],
                form_games_l5=ml_features["form_games_l5"],
                sport=sport,
                features=ml_features,
            )

            # Dynamic Poisson/XGBoost blending based on xG extremity
            if poisson_prob is not None and poisson_prob > 0:
                is_draw = selection == "Draw"
                home_xg = poisson_pred.get("home_xg", 1.35) if poisson_pred else 1.35
                away_xg = poisson_pred.get("away_xg", 1.35) if poisson_pred else 1.35
                xg_diff = abs(home_xg - away_xg)
                xg_bonus = min(0.15, xg_diff * 0.10)

                if is_draw:
                    poisson_w = min(0.75, 0.60 + xg_bonus)
                else:
                    poisson_w = min(0.50, 0.30 + xg_bonus)
                model_p = (1.0 - poisson_w) * model_p + poisson_w * poisson_prob

            # Momentum adjustment
            if abs(market_momentum) > 0.005:
                model_p = max(0.01, min(0.99, model_p + market_momentum * 0.15))

            features[f"{event_id}:{selection}"] = ml_features

            sig = engine.make_signal(
                sport=sport,
                event_id=event_id,
                market="h2h",
                selection=f"{selection} ({home} vs {away})",
                bookmaker_odds=float(target_odds),
                model_probability=model_p,
                source_mode=source_mode,
                reference_book=sharp_book,
                confidence=conf,
                tax_rate=effective_tax,
            )
            signals.append(sig)
            combo_legs.append(
                {
                    "event_id": event_id,
                    "selection": selection,
                    "odds": float(target_odds),
                    "probability": model_p,
                    "sport": sport,
                    "market_type": "h2h",
                }
            )

        # --- Derive Double Chance & Draw No Bet for soccer (from 1X2 sharp odds) ---
        if sport.startswith("soccer") and len(sharp) == 3:
            home_ip = 1.0 / sharp.get(home, 3.0) if sharp.get(home, 3.0) > 1.0 else 0.33
            draw_ip = 1.0 / sharp.get("Draw", 3.0) if sharp.get("Draw", 3.0) > 1.0 else 0.33
            away_ip = 1.0 / sharp.get(away, 3.0) if sharp.get(away, 3.0) > 1.0 else 0.33

            # Double Chance (1X, X2, 12) — derived mathematically
            if "double_chance" in active_markets:
                dc_markets = [
                    ("1X", f"{home} oder Unentschieden", home_ip + draw_ip),
                    ("X2", f"Unentschieden oder {away}", draw_ip + away_ip),
                    ("12", f"{home} oder {away}", home_ip + away_ip),
                ]
                for dc_code, dc_label, dc_prob in dc_markets:
                    if dc_prob > 0.01 and dc_prob < 1.0:
                        dc_odds = round(1.0 / dc_prob, 2)
                        if dc_odds >= min_odds_threshold:
                            sig = engine.make_signal(
                                sport=sport,
                                event_id=event_id,
                                market=f"double_chance {dc_code}",
                                selection=f"{dc_label} ({home} vs {away})",
                                bookmaker_odds=dc_odds,
                                model_probability=min(0.99, dc_prob),
                                source_mode=source_mode,
                                reference_book=sharp_book,
                                confidence=conf * 0.9,
                                tax_rate=effective_tax,
                            )
                            signals.append(sig)
                            combo_legs.append({
                                "event_id": event_id,
                                "selection": dc_label,
                                "odds": dc_odds,
                                "probability": min(0.99, dc_prob),
                                "sport": sport,
                                "market_type": "double_chance",
                            })

            # Draw No Bet (DNB) — remove draw, redistribute
            if "draw_no_bet" in active_markets:
                total_no_draw = home_ip + away_ip
                if total_no_draw > 0.01:
                    dnb_markets = [
                        (home, home_ip / total_no_draw),
                        (away, away_ip / total_no_draw),
                    ]
                    for dnb_name, dnb_prob in dnb_markets:
                        if dnb_prob > 0.01 and dnb_prob < 1.0:
                            dnb_odds = round(1.0 / dnb_prob, 2)
                            if dnb_odds >= min_odds_threshold:
                                sig = engine.make_signal(
                                    sport=sport,
                                    event_id=event_id,
                                    market="draw_no_bet",
                                    selection=f"DNB {dnb_name} ({home} vs {away})",
                                    bookmaker_odds=dnb_odds,
                                    model_probability=min(0.99, dnb_prob),
                                    source_mode=source_mode,
                                    reference_book=sharp_book,
                                    confidence=conf * 0.85,
                                    tax_rate=effective_tax,
                                )
                                signals.append(sig)
                                combo_legs.append({
                                    "event_id": event_id,
                                    "selection": f"DNB {dnb_name}",
                                    "odds": dnb_odds,
                                    "probability": min(0.99, dnb_prob),
                                    "sport": sport,
                                    "market_type": "draw_no_bet",
                                })

        # --- Process spreads market ---
        if "spreads" in active_markets:
            spreads_prices = _extract_prices(e.get("bookmakers") or [], market_key="spreads")
            sharp_spreads_book, sharp_spreads = _pick_sharp(spreads_prices)
            target_spreads_book, target_spreads = _pick_target_book(spreads_prices, sharp_spreads_book) if sharp_spreads_book else ("", {})

            if target_spreads_book and target_spreads and sharp_spreads:
                for sel_label, t_odds in target_spreads.items():
                    if float(t_odds) < min_odds_threshold:
                        continue
                    s_odds = sharp_spreads.get(sel_label)
                    if not s_odds:
                        continue
                    parts = sel_label.split("|")
                    sel_name = parts[0]
                    point_val = float(parts[1]) if len(parts) > 1 else 0.0
                    sharp_prob_sp = 1.0 / s_odds if s_odds > 1.0 else 0.5

                    # Adjust spreads with momentum for NBA/NFL
                    sp_momentum = sport_momentum.get(event_id, {}).get(sel_name, sharp_prob_sp)
                    sp_momentum_delta = sharp_prob_sp - sp_momentum
                    model_p_sp = sharp_prob_sp
                    if abs(sp_momentum_delta) > 0.005 and sport in ("basketball_nba", "americanfootball_nfl"):
                        model_p_sp = max(0.01, min(0.99, sharp_prob_sp + sp_momentum_delta * 0.15))

                    sig = engine.make_signal(
                        sport=sport,
                        event_id=event_id,
                        market=f"spreads {point_val:+.1f}",
                        selection=f"{sel_name} {point_val:+.1f} ({home} vs {away})",
                        bookmaker_odds=float(t_odds),
                        model_probability=min(0.99, max(0.01, model_p_sp)),
                        source_mode=source_mode,
                        reference_book=sharp_spreads_book,
                        confidence=conf * 0.8,
                        tax_rate=effective_tax,
                    )
                    signals.append(sig)
                    combo_legs.append({
                        "event_id": event_id,
                        "selection": f"{sel_name} {point_val:+.1f}",
                        "odds": float(t_odds),
                        "probability": min(0.99, max(0.01, model_p_sp)),
                        "sport": sport,
                        "market_type": "spreads",
                    })

        # --- Process totals market ---
        if "totals" in active_markets:
            totals_prices = _extract_prices(e.get("bookmakers") or [], market_key="totals")
            sharp_totals_book, sharp_totals = _pick_sharp(totals_prices)
            target_totals_book, target_totals = _pick_target_book(totals_prices, sharp_totals_book) if sharp_totals_book else ("", {})

            if target_totals_book and target_totals and sharp_totals:
                for sel_label, t_odds in target_totals.items():
                    if float(t_odds) < min_odds_threshold:
                        continue
                    s_odds = sharp_totals.get(sel_label)
                    if not s_odds:
                        continue
                    parts = sel_label.split("|")
                    sel_name = parts[0]  # "Over" or "Under"
                    point_val = float(parts[1]) if len(parts) > 1 else 2.5
                    sharp_prob_tot = 1.0 / s_odds if s_odds > 1.0 else 0.5

                    # Dynamic Poisson blend for soccer totals — heavy weight (0.70 base)
                    # Uses expanded Poisson keys: over_0_5, over_1_5, over_2_5, over_3_5
                    if poisson_pred and sport.startswith("soccer") and sel_name in ("Over", "Under"):
                        # Map point value to Poisson key
                        poisson_key_map = {
                            0.5: ("over_0_5", "under_0_5"),
                            1.5: ("over_1_5", "under_1_5"),
                            2.5: ("over_2_5", "under_2_5"),
                            3.5: ("over_3_5", "under_3_5"),
                        }
                        keys = poisson_key_map.get(point_val, ("over_2_5", "under_2_5"))
                        poisson_key = keys[0] if sel_name == "Over" else keys[1]
                        poisson_ou = poisson_pred.get(poisson_key, sharp_prob_tot)
                        total_xg = poisson_pred.get("home_xg", 1.35) + poisson_pred.get("away_xg", 1.35)
                        xg_dev = abs(total_xg - 2.7)
                        xg_bonus = min(0.15, xg_dev * 0.10)
                        # Heavy Poisson weight for totals: base 0.70 (up from 0.60)
                        poisson_w = min(0.85, 0.70 + xg_bonus)
                        model_p_tot = (1.0 - poisson_w) * sharp_prob_tot + poisson_w * poisson_ou
                    else:
                        model_p_tot = sharp_prob_tot

                    sig = engine.make_signal(
                        sport=sport,
                        event_id=event_id,
                        market=f"totals {point_val}",
                        selection=f"{sel_name} {point_val} ({home} vs {away})",
                        bookmaker_odds=float(t_odds),
                        model_probability=min(0.99, max(0.01, model_p_tot)),
                        source_mode=source_mode,
                        reference_book=sharp_totals_book,
                        confidence=conf * 0.85,
                        tax_rate=effective_tax,
                    )
                    signals.append(sig)
                    combo_legs.append({
                        "event_id": event_id,
                        "selection": f"{sel_name} {point_val}",
                        "odds": float(t_odds),
                        "probability": min(0.99, max(0.01, model_p_tot)),
                        "sport": sport,
                        "market_type": "totals",
                    })

    # Rank by hit probability (70%) + positive EV (30%), respecting min odds
    ranked = [s for s in signals if s.expected_value > 0]
    ranked.sort(
        key=lambda s: (s.model_probability * 0.7 + min(s.expected_value * 10, 1.0) * 0.3),
        reverse=True,
    )
    top10 = ranked[:10]

    # Build combos for configured sizes (default: 10, 20, 30)
    from src.core.combo_optimizer import ComboOptimizer
    optimizer = ComboOptimizer(engine)
    eligible_legs = [l for l in combo_legs if l["probability"] >= 0.50]
    combos = optimizer.build_all_combos(eligible_legs, target_sizes=combo_sizes)

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "count": len(top10),
        "items": [r.model_dump() for r in top10],
    }
    stats["signals"] = len(top10)

    cache.set_json(SNAPSHOT_KEY, payload, ttl_seconds=12 * 3600)
    cache.set_json(COMBO_KEY, combos, ttl_seconds=12 * 3600)
    cache.set_json(COMBO_LEGS_KEY, combo_legs, ttl_seconds=12 * 3600)
    cache.set_json(META_KEY, stats, ttl_seconds=12 * 3600)

    try:
        auto_place_virtual_bets(top10, features)
    except Exception:
        pass

    return top10


def get_cached_signals() -> Tuple[List[Dict[str, Any]], str]:
    snap = cache.get_json(SNAPSHOT_KEY) or {}
    return snap.get("items") or [], snap.get("ts") or ""


def get_cached_combos() -> List[Dict[str, Any]]:
    return cache.get_json(COMBO_KEY) or []


def get_cached_combo_legs() -> List[Dict[str, Any]]:
    return cache.get_json(COMBO_LEGS_KEY) or []


def get_cached_meta() -> Dict[str, Any]:
    return cache.get_json(META_KEY) or {}
