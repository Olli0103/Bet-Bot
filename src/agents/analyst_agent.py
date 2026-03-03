"""Analyst Agent: Deep event analysis triggered by Scout alerts.

When the Scout detects a steam move or breaking injury, the Analyst
performs full analysis: news enrichment, sentiment, injury aggregation,
ML model prediction, EV computation, and optional LLM reasoning.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from src.core.betting_math import expected_value, public_bias_score
from src.core.elo_ratings import EloSystem
from src.core.enrichment import team_sentiment_score
from src.core.feature_engineering import FeatureEngineer
from src.core.form_tracker import get_form_l5
from src.core.h2h_tracker import get_h2h_features
from src.core.poisson_model import PoissonSoccerModel
from src.core.pricing_model import QuantPricingModel
from src.core.settings import settings
from src.core.volatility_tracker import get_volatility_features

log = logging.getLogger(__name__)


class AnalystAgent:
    """Deeply analyzes an event when triggered by the Scout."""

    def __init__(self) -> None:
        self.qpm = QuantPricingModel()
        self.elo = EloSystem()
        self.poisson = PoissonSoccerModel()

    async def analyze_event(
        self,
        event_id: str,
        sport: str,
        home: str,
        away: str,
        selection: str,
        target_odds: float,
        sharp_odds: float,
        sharp_market: Dict[str, float],
        trigger: str = "manual",
        market_momentum: float = 0.0,
    ) -> Dict[str, Any]:
        """Full analysis of a single event/selection.

        Returns a dict with model probability, EV, features, and recommendation.
        """
        result: Dict[str, Any] = {
            "event_id": event_id,
            "sport": sport,
            "home": home,
            "away": away,
            "selection": selection,
            "trigger": trigger,
        }

        # 1. Sentiment (run in thread to avoid blocking the event loop)
        try:
            sent_home = await asyncio.to_thread(team_sentiment_score, home)
            sent_away = await asyncio.to_thread(team_sentiment_score, away)
        except Exception:
            sent_home = sent_away = 0.0

        # 2. Injury aggregation (API-Sports + Rotowire RSS + LLM)
        inj_home = inj_away = 0
        injury_penalty_home = 0.0
        injury_penalty_away = 0.0
        injury_details: list = []
        try:
            from src.integrations.injury_aggregator import (
                aggregate_injury_intel,
                get_injury_impact_score,
            )
            inj_result = await aggregate_injury_intel(home, away, sport)
            injury_details = inj_result.get("injuries", [])

            # Count injuries per team for feature engineering
            for inj in injury_details:
                team = (inj.get("team") or "").lower()
                if team and (home.lower() in team or team in home.lower()):
                    inj_home += 1
                elif team and (away.lower() in team or team in away.lower()):
                    inj_away += 1

            # Compute impact scores for EV penalty
            injury_penalty_home = get_injury_impact_score(injury_details, home)
            injury_penalty_away = get_injury_impact_score(injury_details, away)
        except Exception as exc:
            log.warning("Injury aggregation in analyst failed: %s", exc)

        # 3. Form
        try:
            home_wr, home_gp = get_form_l5(home)
            away_wr, away_gp = get_form_l5(away)
        except Exception:
            home_wr, home_gp = 0.5, 0
            away_wr, away_gp = 0.5, 0

        # 4. Elo
        try:
            elo_feats = self.elo.get_elo_features(home, away)
        except Exception:
            elo_feats = {"elo_diff": 0.0, "elo_expected": 0.5}

        # 5. H2H
        try:
            h2h_feats = get_h2h_features(home, away)
        except Exception:
            h2h_feats = {"h2h_home_winrate": 0.5}

        # 6. Volatility
        try:
            vol_feats = get_volatility_features(home, away)
        except Exception:
            vol_feats = {"home_volatility": 0.0, "away_volatility": 0.0}

        # 7. Poisson (soccer)
        poisson_prob = None
        poisson_pred = None
        if sport.startswith("soccer"):
            try:
                poisson_pred = self.poisson.predict_match(home, away)
                is_home_sel = selection == home
                if is_home_sel:
                    poisson_prob = poisson_pred.get("h2h_home")
                elif selection == "Draw":
                    poisson_prob = poisson_pred.get("h2h_draw")
                else:
                    poisson_prob = poisson_pred.get("h2h_away")
            except Exception:
                pass

        is_home = selection == home
        sel_wr = home_wr if is_home else away_wr
        sel_gp = home_gp if is_home else away_gp

        # 7b. Stats-based features (Phase 4)
        home_snap: Dict[str, Any] = {}
        away_snap: Dict[str, Any] = {}
        try:
            from src.core.stats_ingester import get_event_snapshot
            home_snap = get_event_snapshot(event_id, home) or {}
            away_snap = get_event_snapshot(event_id, away) or {}
        except Exception:
            pass

        sel_snap = home_snap if is_home else away_snap
        opp_snap = away_snap if is_home else home_snap

        home_away_split_delta = 0.0
        if sel_snap.get("home_win_rate") is not None and sel_snap.get("away_win_rate") is not None:
            home_away_split_delta = (sel_snap["home_win_rate"] - sel_snap["away_win_rate"])

        league_position_delta = 0.0
        if sel_snap.get("league_position") and opp_snap.get("league_position"):
            league_position_delta = float(opp_snap["league_position"] - sel_snap["league_position"])

        # 8. Public bias detection (Tipico market shading vs sharp)
        bias = public_bias_score(sharp_market, {selection: target_odds})
        sel_bias = bias.get(selection, 0.0)

        # 9. Build features (including market momentum + Phase 4 stats)
        ml_features = FeatureEngineer.build_core_features(
            target_odds=target_odds,
            sharp_odds=sharp_odds,
            sharp_market=sharp_market,
            sentiment_home=sent_home,
            sentiment_away=sent_away,
            injuries_home=inj_home,
            injuries_away=inj_away,
            selection=selection,
            home_team=home,
            form_winrate_l5=sel_wr,
            form_games_l5=sel_gp,
            elo_diff=elo_feats.get("elo_diff", 0.0),
            elo_expected=elo_feats.get("elo_expected", 0.5),
            h2h_home_winrate=h2h_feats.get("h2h_home_winrate", 0.5),
            home_volatility=vol_feats.get("home_volatility", 0.0),
            away_volatility=vol_feats.get("away_volatility", 0.0),
            poisson_true_prob=poisson_prob,
            public_bias=sel_bias,
            market_momentum=market_momentum,
            team_attack_strength=sel_snap.get("attack_strength", 1.0),
            team_defense_strength=sel_snap.get("defense_strength", 1.0),
            opp_attack_strength=opp_snap.get("attack_strength", 1.0),
            opp_defense_strength=opp_snap.get("defense_strength", 1.0),
            form_trend_slope=sel_snap.get("form_trend_slope", 0.0),
            rest_days=sel_snap.get("rest_days"),
            schedule_congestion=sel_snap.get("schedule_congestion", 0.0),
            over25_rate=sel_snap.get("over25_rate", 0.0),
            btts_rate=sel_snap.get("btts_rate", 0.0),
            home_away_split_delta=home_away_split_delta,
            league_position_delta=league_position_delta,
            goals_scored_avg=sel_snap.get("goals_scored_avg", 0.0),
            goals_conceded_avg=sel_snap.get("goals_conceded_avg", 0.0),
        )

        # 10. Model prediction
        model_p = self.qpm.get_true_probability(
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

        # Momentum adjustment: nudge model_p toward market direction
        if abs(market_momentum) > 0.005:
            model_p = model_p + market_momentum * 0.15
            model_p = max(0.01, min(0.99, model_p))

        # 11. Injury penalty: if key players are confirmed Out for the selected
        # team, apply a negative confidence adjustment to model_p.
        # Both penalty values are <= 0 (negative impact scores).
        sel_injury_penalty = injury_penalty_home if is_home else injury_penalty_away
        opp_injury_penalty = injury_penalty_away if is_home else injury_penalty_home
        # Net effect: injuries on selected team hurt (negative), opponent injuries help (positive).
        # sel_injury_penalty is <= 0 → makes net negative (hurts selected team)
        # opp_injury_penalty is <= 0 → subtracting it makes net positive (helps selected team)
        net_injury_effect = sel_injury_penalty - opp_injury_penalty
        # Apply as a direct probability adjustment (clamped)
        if abs(net_injury_effect) > 0.01:
            model_p = max(0.01, min(0.99, model_p + net_injury_effect * 0.50))

        # 12. EV calculation
        tax_rate = settings.tipico_tax_rate if not settings.tax_free_mode else 0.0
        ev = expected_value(model_p, target_odds, tax_rate=tax_rate)

        # Public bias skepticism: if Tipico is shading this favorite heavily,
        # raise the EV threshold for a BET recommendation
        ev_threshold = 0.01
        if sel_bias > 0.02:
            ev_threshold = 0.02  # require stronger edge on shaded favorites

        result.update({
            "model_probability": round(model_p, 4),
            "expected_value": round(ev, 4),
            "features": ml_features,
            "sentiment": {"home": sent_home, "away": sent_away},
            "injuries": {"home": inj_home, "away": inj_away},
            "injury_details": injury_details,
            "injury_penalty": {"home": injury_penalty_home, "away": injury_penalty_away},
            "form": {"home_wr": home_wr, "away_wr": away_wr},
            "elo": elo_feats,
            "poisson_prob": poisson_prob,
            "public_bias": sel_bias,
            # Preserve input parameters so downstream (cache, Deep Dive) can
            # reproduce identical results without re-guessing inputs.
            "sharp_odds": sharp_odds,
            "sharp_market": sharp_market,
            "market_momentum": market_momentum,
            "bookmaker_odds": target_odds,
            "recommendation": "BET" if ev > ev_threshold else "SKIP",
        })

        return result

    async def reason_with_llm(self, context: Dict) -> Optional[str]:
        """LLM reasoning optimized for Gemma 3 4B.

        Uses concise German prompts with strict data-only reasoning.
        """
        try:
            from src.integrations.ollama_sentiment import OllamaSentimentClient
            nlp = OllamaSentimentClient()

            model_p = context.get("model_probability", 0)
            ev = context.get("expected_value", 0)
            momentum = context.get("market_momentum", 0.0)

            # Include injury context if available
            inj_details = context.get("injury_details", [])
            inj_text = ""
            if inj_details:
                inj_lines = [f"  {i['player']} ({i['team']}): {i['status']}" for i in inj_details[:5]]
                inj_text = f"\nVerletzungen:\n" + "\n".join(inj_lines)

            prompt = (
                "Basiere deine Analyse STRIKT auf den Daten. "
                "Erstelle eine ultrakurze, praegnante Zusammenfassung (maximal 30 Woerter) auf Deutsch. "
                "Verwende KEINE Formatierungen wie Sterne, fett oder kursiv. Antworte in reinem Text.\n\n"
                f"Sport: {context.get('sport')}\n"
                f"Match: {context.get('home')} vs {context.get('away')}\n"
                f"Tipp: {context.get('selection')}\n"
                f"Modell-Wk: {model_p:.0%}\n"
                f"Erwarteter Profit (EV): {ev * 100:+.2f}%\n"
                f"Wettmarkt-Momentum: {momentum * 100:+.1f}%\n"
                f"Empfehlung: {'Wetten' if ev > 0.01 else 'Nicht wetten'}"
                f"{inj_text}"
            )

            result = await asyncio.to_thread(nlp.generate_raw, prompt)
            return result if result else None
        except Exception:
            return None
