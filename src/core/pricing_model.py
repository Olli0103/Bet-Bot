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
    if sport_key.startswith("americanfootball"):
        return "americanfootball"
    if sport_key.startswith("icehockey"):
        return "icehockey"
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
        if Path(weights_file).exists():
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
        market: str = "h2h",
    ) -> float:
        """Get true probability using the best available model.

        Parameters
        ----------
        features : dict, optional
            Full feature dict from FeatureEngineer.build_core_features().
            If provided, attempts XGBoost prediction first.
        sport : str
            Sport key for selecting the sport-specific model.
        market : str
            Market type for calibration lookup.

        When calibration is enabled, the raw model output is passed through
        the CalibrationManager and the calibrated probability is returned.
        The raw probability is stored as ``_last_raw_prob`` and the
        calibration source as ``_last_calibration_source`` for diagnostic
        logging by the caller.
        """
        # Try XGBoost model if we have the full feature dict
        if features is not None:
            prob = self._xgboost_predict(features, sport)
            if prob is not None:
                raw = prob
            else:
                raw = self._legacy_predict(
                    sharp_prob=sharp_prob,
                    sentiment=sentiment,
                    injuries=injuries,
                    clv=clv,
                    sharp_vig=sharp_vig,
                    form_winrate_l5=form_winrate_l5,
                    form_games_l5=form_games_l5,
                )
        else:
            raw = self._legacy_predict(
                sharp_prob=sharp_prob,
                sentiment=sentiment,
                injuries=injuries,
                clv=clv,
                sharp_vig=sharp_vig,
                form_winrate_l5=form_winrate_l5,
                form_games_l5=form_games_l5,
            )

        # Store raw for diagnostic access
        self._last_raw_prob = raw
        self._last_calibration_source = "raw_passthrough"

        # Apply calibration layer
        from src.core.settings import settings as _s
        if _s.calibration_enabled:
            try:
                from src.core.calibration import get_calibration_manager
                mgr = get_calibration_manager(_s.calibration_method)
                calibrated, source = mgr.calibrate(raw, sport, market)

                # Post-calibration shrinkage toward market-implied probability
                # to reduce overconfident tails that cause Kelly cap-banging.
                # This is applied only for h2h where sharp_prob is reliable.
                if market == "h2h":
                    try:
                        market_anchor = float(sharp_prob)
                        if 0.01 <= market_anchor <= 0.99:
                            shrink = 0.35
                            calibrated = calibrated * (1.0 - shrink) + market_anchor * shrink
                    except Exception:
                        pass

                calibrated = max(0.01, min(0.99, calibrated))
                self._last_calibration_source = source
                return calibrated
            except Exception as exc:
                log.debug("Calibration failed, using raw: %s", exc)

        return raw

    def _xgboost_predict(self, features: Dict[str, float], sport: str) -> Optional[float]:
        """Try to predict using sport-specific or general XGBoost model.

        Quality checks:
        - Rejects models with Brier score > 0.25 (worse than random)
        - Falls back to general if sport-specific model is unreliable
        - Applies reliability adjustment from calibration bins
        - Blends with CLV regressor output when available
        """
        group = _sport_group(sport)

        # Try sport-specific model first, then general
        classifier_prob = None
        for model_group in [group, "general"]:
            model_data = _load_joblib_model(model_group)
            if model_data is None:
                continue

            model = model_data.get("model")
            model_features = model_data.get("features", [])
            if model is None or not model_features:
                continue

            # Quality gate: skip models with poor calibration
            metrics = model_data.get("metrics", {})
            brier = metrics.get("brier_score", 1.0)
            if brier > 0.25:
                log.warning(
                    "Skipping model %s: Brier=%.4f > 0.25 (unreliable)",
                    model_group, brier,
                )
                continue

            # Minimum sample check
            n_samples = model_data.get("n_samples", 0)
            if n_samples < 50:
                log.warning(
                    "Skipping model %s: only %d samples (need 50+)",
                    model_group, n_samples,
                )
                continue

            try:
                x = np.array([[float(features.get(f, 0.0)) for f in model_features]])
                prob = float(model.predict_proba(x)[0, 1])
                classifier_prob = max(0.01, min(0.99, prob))

                # Apply reliability adjustment from calibration bins
                rel_adj = self._get_reliability_adjustment(classifier_prob, model_data)
                if rel_adj != 1.0:
                    sharp_p = features.get("sharp_implied_prob", classifier_prob)
                    classifier_prob = classifier_prob * rel_adj + sharp_p * (1.0 - rel_adj)
                    classifier_prob = max(0.01, min(0.99, classifier_prob))

                break  # Got a valid classifier prediction
            except Exception as exc:
                log.debug("XGBoost predict failed (%s): %s", model_group, exc)
                continue

        if classifier_prob is None:
            return None

        # Blend with CLV regressor if available
        clv_prob = self._clv_predict(features)
        if clv_prob is not None:
            # Inverse-variance weighting with metric-scale correction.
            #
            # Problem: Brier score (classifier error, range [0,~0.25]) and
            # CLV MSE (regressor error against continuous target, often
            # much smaller ~0.01-0.05) are on fundamentally different
            # scales.  Raw inverse-variance would let the CLV dominate.
            #
            # Solution: scale CLV MSE by 4.0 to approximate Brier-score
            # magnitude.  Brier = E[(p - y)^2] with binary y in {0,1},
            # CLV MSE = E[(p - closing_prob)^2] with closing_prob in [0,1].
            # The scaling factor compensates for the variance reduction
            # when predicting against a continuous target vs binary labels.
            eps = 1e-6
            var_classifier = model_data.get("metrics", {}).get("brier_score", 0.25)
            clv_model_data = _load_joblib_model("clv_general")
            raw_clv_mse = (
                clv_model_data.get("metrics", {}).get("clv_mse", 0.05)
                if clv_model_data else 0.05
            )
            var_clv_scaled = raw_clv_mse * 4.0

            w_classifier_raw = 1.0 / (var_classifier + eps)
            w_clv_raw = 1.0 / (var_clv_scaled + eps)
            w_classifier = w_classifier_raw / (w_classifier_raw + w_clv_raw)

            # Clamp CLV weight to [0.10, 0.40] to prevent extreme allocations
            w_clv = 1.0 - w_classifier
            w_clv = max(0.10, min(0.40, w_clv))
            w_classifier = 1.0 - w_clv

            blended = w_classifier * classifier_prob + w_clv * clv_prob
            log.debug(
                "Blending: w_classifier=%.2f (brier=%.4f) w_clv=%.2f (mse=%.4f, scaled=%.4f) -> %.4f",
                w_classifier, var_classifier, w_clv, raw_clv_mse, var_clv_scaled, blended,
            )
            return max(0.01, min(0.99, blended))

        return classifier_prob

    def _clv_predict(self, features: Dict[str, float]) -> Optional[float]:
        """Predict sharp closing probability using the CLV regressor."""
        model_data = _load_joblib_model("clv_general")
        if model_data is None:
            return None

        model = model_data.get("model")
        model_features = model_data.get("features", [])
        if model is None or not model_features:
            return None

        # Quality gate: CLV model must beat its baseline
        metrics = model_data.get("metrics", {})
        if metrics.get("clv_mse", 1.0) >= metrics.get("clv_mse_baseline", 1.0):
            return None

        try:
            x = np.array([[float(features.get(f, 0.0)) for f in model_features]])
            pred = float(model.predict(x)[0])
            return max(0.01, min(0.99, pred))
        except Exception as exc:
            log.debug("CLV regressor predict failed: %s", exc)
            return None

    @staticmethod
    def _get_reliability_adjustment(prob: float, model_data: Dict) -> float:
        """Return reliability adjustment factor from calibration bins."""
        bins = model_data.get("metrics", {}).get("reliability_bins", {})
        if not bins:
            return 1.0

        for bin_key, info in bins.items():
            try:
                low, high = (float(x) for x in bin_key.split("_"))
            except (ValueError, TypeError):
                continue
            if low <= prob < high:
                adj = float(info.get("kelly_adjustment", 1.0))
                # Cap adjustment to prevent extreme swings
                return max(0.5, min(1.5, adj))

        return 1.0
