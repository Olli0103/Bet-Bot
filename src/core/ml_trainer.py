"""XGBoost-based ML training pipeline with calibration and sport-specific models.

Replaces the original LogisticRegression approach with:
- XGBClassifier with tuned hyperparameters
- CalibratedClassifierCV (isotonic, 3-fold) for probability calibration
- TimeSeriesSplit for proper temporal cross-validation
- Sport-specific model training when enough samples exist
- Post-training validation (Brier score, calibration, feature importance)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select
from xgboost import XGBClassifier

from src.data.models import PlacedBet
from src.data.postgres import SessionLocal

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")

# Full feature set (Phase 1-3 + review enhancements)
FEATURES = [
    "sentiment_delta",
    "injury_delta",
    "sharp_implied_prob",
    "clv",
    "sharp_vig",
    "form_winrate_l5",
    "form_games_l5",
    "elo_diff",
    "elo_expected",
    "h2h_home_winrate",
    "home_advantage",
    "weather_rain",
    "weather_wind_high",
    "home_volatility",
    "away_volatility",
    "is_steam_move",
    "line_staleness",
    "twitter_sentiment_delta",
    "time_to_kickoff_hours",
    "public_bias",
]

# Legacy features (backward-compatible with old JSON weights)
LEGACY_FEATURES = [
    "sentiment_delta",
    "injury_delta",
    "sharp_implied_prob",
    "clv",
    "sharp_vig",
    "form_winrate_l5",
    "form_games_l5",
]

EPS = 1e-12

# Sport groupings for sport-specific models
SPORT_GROUPS: Dict[str, str] = {}
for prefix in ("soccer_", "football_"):
    for suffix in (
        "epl", "germany_bundesliga", "spain_la_liga", "italy_serie_a",
        "france_ligue_one", "uefa_champs_league", "uefa_europa_league",
        "germany_bundesliga2", "england_league1", "england_league2",
    ):
        SPORT_GROUPS[f"{prefix}{suffix}"] = "soccer"
for prefix in ("basketball_",):
    for suffix in ("nba", "euroleague", "germany_bbl", "spain_acb"):
        SPORT_GROUPS[f"{prefix}{suffix}"] = "basketball"
for prefix in ("tennis_",):
    for suffix in ("atp", "wta", "atp_french_open", "atp_wimbledon", "atp_us_open", "atp_australian_open"):
        SPORT_GROUPS[f"{prefix}{suffix}"] = "tennis"


def _get_sport_group(sport_key: str) -> str:
    """Map a sport key to its group (soccer, basketball, tennis, or 'general')."""
    if sport_key in SPORT_GROUPS:
        return SPORT_GROUPS[sport_key]
    for prefix, group in [("soccer", "soccer"), ("basketball", "basketball"), ("tennis", "tennis")]:
        if sport_key.startswith(prefix):
            return group
    return "general"


def _model_path(sport_group: str) -> Path:
    return MODELS_DIR / f"xgb_{sport_group}.joblib"


def _clean_frame(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in feature_list:
        if c not in out.columns:
            out[c] = 0.0
    out[feature_list] = out[feature_list].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # clamp outliers
    if "sentiment_delta" in out.columns:
        out["sentiment_delta"] = out["sentiment_delta"].clip(-5.0, 5.0)
    if "injury_delta" in out.columns:
        out["injury_delta"] = out["injury_delta"].clip(-20.0, 20.0)
    if "sharp_implied_prob" in out.columns:
        out["sharp_implied_prob"] = out["sharp_implied_prob"].clip(0.0, 1.0)
    if "clv" in out.columns:
        out["clv"] = out["clv"].clip(-5.0, 5.0)
    if "sharp_vig" in out.columns:
        out["sharp_vig"] = out["sharp_vig"].clip(-0.2, 0.5)
    if "form_winrate_l5" in out.columns:
        out["form_winrate_l5"] = out["form_winrate_l5"].clip(0.0, 1.0)
    if "form_games_l5" in out.columns:
        out["form_games_l5"] = out["form_games_l5"].clip(0.0, 5.0)
    if "time_to_kickoff_hours" in out.columns:
        out["time_to_kickoff_hours"] = out["time_to_kickoff_hours"].clip(0.0, 168.0)
    if "line_staleness" in out.columns:
        out["line_staleness"] = out["line_staleness"].clip(0.0, 120.0)  # cap at 2 hours
    if "public_bias" in out.columns:
        out["public_bias"] = out["public_bias"].clip(-0.5, 0.5)
    return out


def _get_active_features(X: pd.DataFrame, feature_list: List[str]) -> List[str]:
    """Return features with non-zero variance."""
    variances = X[feature_list].var(axis=0)
    return [f for f in feature_list if float(variances.get(f, 0.0)) > EPS]


def _train_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    active_features: List[str],
) -> Tuple[CalibratedClassifierCV, Dict]:
    """Train XGBoost with isotonic calibration and TimeSeriesSplit CV."""
    X_active = X[active_features].values
    y_vals = y.values

    base_model = XGBClassifier(
        objective="binary:logistic",
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    # Calibrate with TimeSeriesSplit
    n_splits = min(5, max(2, len(X_active) // 200))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=tscv,
    )
    calibrated.fit(X_active, y_vals)

    # Evaluate on last fold
    metrics = _evaluate_model(calibrated, X_active, y_vals, tscv)

    return calibrated, metrics


def _evaluate_model(
    model: CalibratedClassifierCV,
    X: np.ndarray,
    y: np.ndarray,
    tscv: TimeSeriesSplit,
) -> Dict:
    """Evaluate model on the last temporal fold with reliability diagram data."""
    metrics = {}
    splits = list(tscv.split(X))
    if splits:
        _, test_idx = splits[-1]
        X_test, y_test = X[test_idx], y[test_idx]
        y_pred = model.predict_proba(X_test)[:, 1]

        metrics["brier_score"] = round(float(brier_score_loss(y_test, y_pred)), 6)
        metrics["log_loss"] = round(float(log_loss(y_test, y_pred)), 6)
        metrics["test_size"] = len(test_idx)

        # Reliability diagram: bin predictions into buckets and compute
        # actual vs predicted rates + adjustment factors for Kelly scaling.
        # If model says 60% but actual is 55%, the Kelly multiplier for
        # that bucket should be scaled down by actual/predicted.
        bins = [0.0, 0.3, 0.45, 0.55, 0.7, 1.0]
        reliability_bins: Dict[str, Dict] = {}
        for i in range(len(bins) - 1):
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
            if mask.sum() > 5:
                actual_rate = float(y_test[mask].mean())
                predicted_rate = float(y_pred[mask].mean())
                # Kelly adjustment: scale stakes by actual/predicted ratio.
                # > 1.0 means model under-predicts (bet more),
                # < 1.0 means model over-predicts (trim stakes).
                kelly_adj = round(actual_rate / max(0.01, predicted_rate), 4)
                bin_key = f"{bins[i]:.2f}_{bins[i+1]:.2f}"
                reliability_bins[bin_key] = {
                    "predicted": round(predicted_rate, 4),
                    "actual": round(actual_rate, 4),
                    "n": int(mask.sum()),
                    "kelly_adjustment": kelly_adj,
                    "over_predicting": predicted_rate > actual_rate + 0.02,
                }
                metrics[f"calib_{bins[i]:.1f}_{bins[i+1]:.1f}"] = {
                    "predicted": round(predicted_rate, 4),
                    "actual": round(actual_rate, 4),
                    "n": int(mask.sum()),
                }

        metrics["reliability_bins"] = reliability_bins
    return metrics


def _validate_model(model, metrics: Dict, active_features: List[str]) -> List[str]:
    """Post-training validation checks."""
    warnings = []

    brier = metrics.get("brier_score", 1.0)
    if brier > 0.25:
        warnings.append(f"Brier score {brier:.4f} > 0.25 (worse than random). Model may be unreliable.")

    ll = metrics.get("log_loss", 1.0)
    if ll > 0.693:
        warnings.append(f"Log loss {ll:.4f} > 0.693 (worse than coin flip).")

    # Check feature importances from the base estimators
    try:
        base_estimators = getattr(model, "calibrated_classifiers_", [])
        if base_estimators:
            importances = base_estimators[0].estimator.feature_importances_
            for i, feat in enumerate(active_features):
                if i < len(importances) and importances[i] < 0.001:
                    warnings.append(f"Feature '{feat}' has near-zero importance ({importances[i]:.4f}).")
    except Exception:
        pass

    return warnings


def auto_train_model(min_samples: int = 500) -> str:
    """Train the general model and also save legacy JSON weights for backward compatibility."""
    MODELS_DIR.mkdir(exist_ok=True)

    with SessionLocal() as db:
        query = select(PlacedBet).where(PlacedBet.status.in_(["won", "lost"]))
        df = pd.read_sql(query, db.bind)

    if len(df) < min_samples:
        return f"Zu wenig Daten: {len(df)}/{min_samples}"

    df = _clean_frame(df, FEATURES)
    X = df[FEATURES]
    y = (df["status"] == "won").astype(int)

    active_features = _get_active_features(X, FEATURES)
    if not active_features:
        return "Abbruch: Keine Feature-Varianz verfügbar."

    model, metrics = _train_xgboost(X, y, active_features)
    warnings = _validate_model(model, metrics, active_features)

    for w in warnings:
        log.warning("ML validation (general): %s", w)

    # Save XGBoost model
    model_data = {
        "model": model,
        "features": active_features,
        "metrics": metrics,
        "n_samples": len(df),
    }
    joblib.dump(model_data, _model_path("general"))
    log.info("Saved general model: %d samples, %d features", len(df), len(active_features))

    # Also save backward-compatible JSON weights (logistic approximation)
    _save_legacy_weights(df, y, active_features, metrics)

    suffix = f" | Warnungen: {'; '.join(warnings)}" if warnings else ""
    brier_str = f"Brier={metrics.get('brier_score', 'N/A')}"
    return f"Erfolg: {len(df)} Wetten trainiert. Aktiv: {', '.join(active_features)} | {brier_str}{suffix}"


def auto_train_all_models(min_samples: int = 200) -> str:
    """Train general + sport-specific XGBoost models."""
    MODELS_DIR.mkdir(exist_ok=True)

    with SessionLocal() as db:
        query = select(PlacedBet).where(PlacedBet.status.in_(["won", "lost"]))
        df = pd.read_sql(query, db.bind)

    if len(df) < min_samples:
        return f"Zu wenig Daten: {len(df)}/{min_samples}"

    df = _clean_frame(df, FEATURES)
    results = []

    # Train general model on all data
    X_all = df[FEATURES]
    y_all = (df["status"] == "won").astype(int)
    active_all = _get_active_features(X_all, FEATURES)

    if active_all:
        model, metrics = _train_xgboost(X_all, y_all, active_all)
        warnings = _validate_model(model, metrics, active_all)
        for w in warnings:
            log.warning("ML validation (general): %s", w)
        joblib.dump({"model": model, "features": active_all, "metrics": metrics, "n_samples": len(df)}, _model_path("general"))
        results.append(f"general: {len(df)} samples, brier={metrics.get('brier_score', 'N/A')}")
        _save_legacy_weights(df, y_all, active_all, metrics)

    # Train sport-specific models
    if "sport" in df.columns:
        df["sport_group"] = df["sport"].apply(_get_sport_group)

        for group in ["soccer", "basketball", "tennis"]:
            subset = df[df["sport_group"] == group]
            if len(subset) < min_samples:
                results.append(f"{group}: skipped ({len(subset)} < {min_samples})")
                continue

            X_sport = subset[FEATURES]
            y_sport = (subset["status"] == "won").astype(int)
            active_sport = _get_active_features(X_sport, FEATURES)

            if not active_sport:
                results.append(f"{group}: no feature variance")
                continue

            model_sport, metrics_sport = _train_xgboost(X_sport, y_sport, active_sport)
            warnings_sport = _validate_model(model_sport, metrics_sport, active_sport)
            for w in warnings_sport:
                log.warning("ML validation (%s): %s", group, w)

            joblib.dump(
                {"model": model_sport, "features": active_sport, "metrics": metrics_sport, "n_samples": len(subset)},
                _model_path(group),
            )
            results.append(f"{group}: {len(subset)} samples, brier={metrics_sport.get('brier_score', 'N/A')}")

    return " | ".join(results)


def _save_legacy_weights(df: pd.DataFrame, y: pd.Series, active_features: List[str], metrics: Dict) -> None:
    """Save backward-compatible JSON weights (logistic regression approximation)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X_legacy = df[[f for f in LEGACY_FEATURES if f in df.columns]]
        active_legacy = [f for f in LEGACY_FEATURES if f in X_legacy.columns and float(X_legacy[f].var()) > EPS]

        if not active_legacy:
            return

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="liblinear")),
        ])
        pipe.fit(X_legacy[active_legacy], y)

        clf = pipe.named_steps["clf"]
        scaler = pipe.named_steps["scaler"]

        coef_std = clf.coef_[0]
        scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
        coef_raw = coef_std / scale
        intercept_raw = float(clf.intercept_[0] - np.sum((scaler.mean_ / scale) * coef_std))

        coef_map = {f: 0.0 for f in LEGACY_FEATURES}
        for i, f in enumerate(active_legacy):
            coef_map[f] = float(coef_raw[i])

        variances = X_legacy.var(axis=0).to_dict()

        learned_weights = {
            **coef_map,
            "intercept": intercept_raw,
            "meta": {
                "samples": int(len(df)),
                "wins": int(y.sum()),
                "losses": int((1 - y).sum()),
                "active_features": active_legacy,
                "variance": {k: float(v) for k, v in variances.items()},
            },
        }

        with open("ml_strategy_weights.json", "w", encoding="utf-8") as f:
            json.dump(learned_weights, f)
    except Exception as exc:
        log.warning("Failed to save legacy weights: %s", exc)


def load_model(sport_group: str = "general") -> Optional[Dict]:
    """Load a trained model from disk. Returns None if not available."""
    path = _model_path(sport_group)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception as exc:
            log.warning("Failed to load model %s: %s", sport_group, exc)
    return None


def get_reliability_adjustment(model_prob: float, sport_group: str = "general") -> float:
    """Return a Kelly multiplier based on calibration reliability bins.

    If the model consistently over-predicts in the bucket containing
    ``model_prob``, the returned multiplier will be < 1.0 to trim stakes.
    If under-predicting, it will be > 1.0. Returns 1.0 when no data.
    """
    model_data = load_model(sport_group)
    if model_data is None:
        model_data = load_model("general")
    if model_data is None:
        return 1.0

    bins = model_data.get("metrics", {}).get("reliability_bins", {})
    if not bins:
        return 1.0

    for bin_key, info in bins.items():
        try:
            low, high = (float(x) for x in bin_key.split("_"))
        except (ValueError, TypeError):
            continue
        if low <= model_prob < high:
            return float(info.get("kelly_adjustment", 1.0))

    return 1.0
