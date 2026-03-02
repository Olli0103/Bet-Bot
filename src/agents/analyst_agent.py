"""Analyst Agent: Deep event analysis triggered by Scout alerts.

When the Scout detects a steam move or breaking injury, the Analyst
performs full analysis: news enrichment, sentiment, injury check,
ML model prediction, EV computation, and optional LLM reasoning.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.core.betting_math import expected_value
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

        # 1. Sentiment
        try:
            sent_home = team_sentiment_score(home)
            sent_away = team_sentiment_score(away)
        except Exception:
            sent_home = sent_away = 0.0

        # 2. Injuries (only for soccer)
        inj_home = inj_away = 0
        if sport.startswith("soccer"):
            try:
                from src.core.enrichment import soccer_injury_delta
                inj_home, inj_away = soccer_injury_delta(home, away, "")
            except Exception:
                pass

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
        if sport.startswith("soccer"):
            try:
                pred = self.poisson.predict_match(home, away)
                is_home = selection == home
                if is_home:
                    poisson_prob = pred.get("h2h_home")
                elif selection == "Draw":
                    poisson_prob = pred.get("h2h_draw")
                else:
                    poisson_prob = pred.get("h2h_away")
            except Exception:
                pass

        is_home = selection == home
        sel_wr = home_wr if is_home else away_wr
        sel_gp = home_gp if is_home else away_gp

        # 8. Build features
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
        )

        # 9. Model prediction
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

        # Blend with Poisson
        if poisson_prob is not None and poisson_prob > 0:
            model_p = 0.7 * model_p + 0.3 * poisson_prob

        # 10. EV calculation
        tax_rate = settings.tipico_tax_rate if not settings.tax_free_mode else 0.0
        ev = expected_value(model_p, target_odds, tax_rate=tax_rate)

        result.update({
            "model_probability": round(model_p, 4),
            "expected_value": round(ev, 4),
            "features": ml_features,
            "sentiment": {"home": sent_home, "away": sent_away},
            "injuries": {"home": inj_home, "away": inj_away},
            "form": {"home_wr": home_wr, "away_wr": away_wr},
            "elo": elo_feats,
            "poisson_prob": poisson_prob,
            "recommendation": "BET" if ev > 0.01 else "SKIP",
        })

        return result

    async def reason_with_llm(self, context: Dict) -> Optional[str]:
        """Optional LLM reasoning about the analysis.

        Uses Ollama (local) or Claude API if configured. Returns a qualitative
        assessment or None if LLM is not available.
        """
        try:
            from src.integrations.ollama_sentiment import OllamaSentimentClient
            nlp = OllamaSentimentClient()

            prompt = (
                f"Analyze this betting opportunity:\n"
                f"Sport: {context.get('sport')}\n"
                f"Match: {context.get('home')} vs {context.get('away')}\n"
                f"Selection: {context.get('selection')}\n"
                f"Model probability: {context.get('model_probability', 0):.2%}\n"
                f"Expected value: {context.get('expected_value', 0):.4f}\n"
                f"Trigger: {context.get('trigger', 'unknown')}\n"
                f"Sentiment: home={context.get('sentiment', {}).get('home', 0):.2f}, "
                f"away={context.get('sentiment', {}).get('away', 0):.2f}\n"
                f"Should we bet? Give a brief assessment."
            )

            result = nlp.analyze(text=prompt, context="betting_analysis")
            return f"LLM: {result.label} ({result.confidence:.2f})"
        except Exception:
            return None
