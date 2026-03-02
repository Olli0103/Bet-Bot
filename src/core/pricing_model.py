"""Pricing model: XGBoost sport-specific models with legacy JSON fallback.

Load order:
1. Sport-specific .joblib model (e.g., models/xgb_soccer.joblib)
2. General .joblib model (models/xgb_general.joblib)
3. Legacy JSON weights (ml_strategy_weights.json) via logistic regression
4. Sharp probability passthrough (no model available)
"""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")

# Cache loaded models in module-level dict to avoid repeated disk reads
_model_cache: Dict[str, Optional[Dict]] = {}


def _load_joblib_model(sport_group: str) -> Optional[Dict]:
    """Load a joblib model, with in-memory caching."""
    if sport_group in _model_cache:
        return _model_cache[sport_group]

    path = MODELS_DIR / f"xgb_{sport_group}.joblib"
    if path.exists():
        try:
            import joblib
            data = joblib.load(path)
            _model_cache[sport_group] = data
            return data
        except Exception as exc:
            log.warning("Failed to load model %s: %s", sport_group, exc)

    _model_cache[sport_group] = None
    return None


def _sport_group(sport_key: str) -> str:
    """Map a sport key to its group."""
    if sport_key.startswith(("soccer", "football")):
        return "soccer"
    if sport_key.startswith("basketball"):
        return "basketball"
    if sport_key.startswith("tennis"):
        return "tennis"
    return "general"


class QuantPricingModel:
    """Unified pricing model with XGBoost + legacy JSON fallback."""

    def __init__(self, weights_file: str = "ml_strategy_weights.json"):
        # Legacy weights for backward compatibility
        self.weights = {
            "sentiment_delta": 0.0,
            "injury_delta": 0.0,
            "sharp_implied_prob": 1.0,
            "clv": 0.0,
            "sharp_vig": 0.0,
            "form_winrate_l5": 0.0,
            "form_games_l5": 0.0,
            "intercept": 0.0,
        }
        if os.path.exists(weights_file):
            try:
                with open(weights_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                for k, v in loaded.items():
                    if k in self.weights:
                        self.weights[k] = float(v)
            except Exception:
                pass

    def _log_odds_to_prob(self, log_odds: float) -> float:
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, log_odds))))

    def _legacy_predict(
        self,
        sharp_prob: float,
        sentiment: float = 0.0,
        injuries: float = 0.0,
        clv: float = 0.0,
        sharp_vig: float = 0.0,
        form_winrate_l5: float = 0.0,
        form_games_l5: float = 0.0,
    ) -> float:
        """Legacy logistic regression prediction using JSON weights."""
        log_odds = float(self.weights.get("intercept", 0.0))
        log_odds += float(sharp_prob) * float(self.weights.get("sharp_implied_prob", 0.0))
        log_odds += float(sentiment) * float(self.weights.get("sentiment_delta", 0.0))
        log_odds += float(injuries) * float(self.weights.get("injury_delta", 0.0))
        log_odds += float(clv) * float(self.weights.get("clv", 0.0))
        log_odds += float(sharp_vig) * float(self.weights.get("sharp_vig", 0.0))
        log_odds += float(form_winrate_l5) * float(self.weights.get("form_winrate_l5", 0.0))
        log_odds += float(form_games_l5) * float(self.weights.get("form_games_l5", 0.0))
        return self._log_odds_to_prob(log_odds)

    def get_true_probability(
        self,
        sharp_prob: float,
        sentiment: float = 0.0,
        injuries: float = 0.0,
        clv: float = 0.0,
        sharp_vig: float = 0.0,
        form_winrate_l5: float = 0.0,
        form_games_l5: float = 0.0,
        sport: str = "",
        features: Optional[Dict[str, float]] = None,
    ) -> float:
        """Get true probability using the best available model.

        Parameters
        ----------
        features : dict, optional
            Full feature dict from FeatureEngineer.build_core_features().
            If provided, attempts XGBoost prediction first.
        sport : str
            Sport key for selecting the sport-specific model.
        """
        # Try XGBoost model if we have the full feature dict
        if features is not None:
            prob = self._xgboost_predict(features, sport)
            if prob is not None:
                return prob

        # Fallback to legacy JSON weights
        return self._legacy_predict(
            sharp_prob=sharp_prob,
            sentiment=sentiment,
            injuries=injuries,
            clv=clv,
            sharp_vig=sharp_vig,
            form_winrate_l5=form_winrate_l5,
            form_games_l5=form_games_l5,
        )

    def _xgboost_predict(self, features: Dict[str, float], sport: str) -> Optional[float]:
        """Try to predict using sport-specific or general XGBoost model."""
        group = _sport_group(sport)

        # Try sport-specific model first, then general
        for model_group in [group, "general"]:
            model_data = _load_joblib_model(model_group)
            if model_data is None:
                continue

            model = model_data.get("model")
            model_features = model_data.get("features", [])
            if model is None or not model_features:
                continue

            try:
                # Build feature vector in the correct order
                x = np.array([[float(features.get(f, 0.0)) for f in model_features]])
                prob = float(model.predict_proba(x)[0, 1])
                return max(0.01, min(0.99, prob))
            except Exception as exc:
                log.debug("XGBoost predict failed (%s): %s", model_group, exc)
                continue

        return None
