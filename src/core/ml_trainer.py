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

# Full feature set (Phase 1-4 + stats-based features)
# NOTE: "clv" (target_odds/sharp_odds - 1) is intentionally excluded.
# It is derived from sharp_implied_prob (already a feature), creates
# circular reasoning in the training set (we only bet when CLV > 0),
# and its name is dangerously close to the post-match closing line
# value stored in sharp_closing_odds/prob.
FEATURES = [
    # Phase 1-3: core + market + enrichment
    "sentiment_delta",
    "injury_delta",
    "sharp_implied_prob",
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
    "injury_news_delta",
    "time_to_kickoff_hours",
    "public_bias",
    "market_momentum",
    "line_velocity",
    # Phase 4: stats-based features (from TeamMatchStats / EventStatsSnapshot)
    "team_attack_strength",
    "team_defense_strength",
    "opp_attack_strength",
    "opp_defense_strength",
    "expected_total_proxy",
    "form_trend_slope",
    "rest_fatigue_score",
    "schedule_congestion",
    "over25_rate",
    "btts_rate",
    "home_away_split_delta",
    "league_position_delta",
    "goals_scored_avg",
    "goals_conceded_avg",
]

# Soccer-specific features (added to FEATURES when training soccer model)
SOCCER_EXTRA_FEATURES = [
    "poisson_true_prob",
]

# Basketball-specific features (future extension)
BASKETBALL_EXTRA_FEATURES: List[str] = []

# Legacy features (backward-compatible with old JSON weights)
LEGACY_FEATURES = [
    "sentiment_delta",
    "injury_delta",
    "sharp_implied_prob",
    "sharp_vig",
    "form_winrate_l5",
    "form_games_l5",
]

EPS = 1e-12

# Semantically correct defaults for features that are missing from the DB.
# Using 0.0 for everything is wrong — e.g. elo_expected=0.0 implies zero
# chance, when the correct neutral value is 0.5 (no Elo advantage).
FEATURE_DEFAULTS: Dict[str, float] = {
    # --- Phase 1: core ---
    # sharp_implied_prob: 0.0 would mean "impossible" — but _clean_frame has
    # a separate fallback that derives it from 1/odds, so 0.0 is fine here
    # (it triggers the derivation logic).
    "form_winrate_l5": 0.5,       # 50% = no form data
    # form_games_l5: 0 is correct (no games = no data)
    # sentiment_delta: 0.0 is correct (no sentiment = neutral)
    # injury_delta: 0.0 is correct (no injury advantage)
    # sharp_vig: 0.0 triggers derivation in _clean_frame, intentional
    # --- Phase 2: market + enrichment ---
    "elo_expected": 0.5,           # 50% = no Elo advantage
    # elo_diff: 0.0 is correct (no difference)
    "h2h_home_winrate": 0.5,       # 50% = no H2H data
    "home_advantage": 0.5,         # unknown venue
    "time_to_kickoff_hours": 24.0,  # 0.0 would mean "match already started"
    # weather_rain: 0.0 is correct (assume dry)
    # weather_wind_high: 0.0 is correct (assume calm)
    # home_volatility / away_volatility: 0.0 is correct (no line movement)
    # is_steam_move: 0.0 is correct (no steam detected)
    # line_staleness: 0.0 is correct (assume fresh lines)
    # injury_news_delta: 0.0 is correct (no news)
    # public_bias: 0.0 is correct (balanced)
    # market_momentum: 0.0 is correct (no momentum)
    # line_velocity: 0.0 is correct (no movement)
    # --- Phase 4: stats-based ---
    "team_attack_strength": 1.0,   # 1.0 = league average
    "team_defense_strength": 1.0,
    "opp_attack_strength": 1.0,
    "opp_defense_strength": 1.0,
    "expected_total_proxy": 2.7,   # neutral: 1.0 * 1.0 * 1.35 * 2
    "rest_fatigue_score": 0.3,     # 0.0 = "fully rested" is too optimistic
    "schedule_congestion": 0.15,   # ~1 game/week is typical
    "over25_rate": 0.55,           # ~55% of soccer matches have >2.5 goals
    "btts_rate": 0.50,             # ~50% both teams score
    "goals_scored_avg": 1.35,      # league average (soccer)
    "goals_conceded_avg": 1.35,    # league average (soccer)
    # home_away_split_delta: 0.0 is correct (no difference)
    # league_position_delta: 0.0 is correct (equal positions)
    # poisson_true_prob: 0.0 triggers XGBoost native missing handling
}

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
for prefix in ("americanfootball_",):
    for suffix in ("nfl", "ncaaf"):
        SPORT_GROUPS[f"{prefix}{suffix}"] = "americanfootball"
for prefix in ("icehockey_",):
    for suffix in ("nhl",):
        SPORT_GROUPS[f"{prefix}{suffix}"] = "icehockey"


def _get_sport_group(sport_key: str) -> str:
    """Map a sport key to its group (soccer, basketball, tennis, or 'general')."""
    if sport_key in SPORT_GROUPS:
        return SPORT_GROUPS[sport_key]
    for prefix, group in [
        ("soccer", "soccer"), ("football", "soccer"),
        ("basketball", "basketball"),
        ("tennis", "tennis"),
        ("americanfootball", "americanfootball"),
        ("icehockey", "icehockey"),
    ]:
        if sport_key.startswith(prefix):
            return group
    return "general"


def _model_path(sport_group: str) -> Path:
    return MODELS_DIR / f"xgb_{sport_group}.joblib"


def _clean_frame(df: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    out = df.copy()

    # Unpack meta_features JSONB blob — this is where Phase 2-4 features
    # are stored since they don't have dedicated DB columns.
    if "meta_features" in out.columns:
        meta_rows = out["meta_features"].dropna()
        if len(meta_rows) > 0:
            meta_df = pd.json_normalize(meta_rows)
            meta_df.index = meta_rows.index
            for col in meta_df.columns:
                if col in feature_list and col not in out.columns:
                    out[col] = np.nan
                    out.loc[meta_df.index, col] = pd.to_numeric(
                        meta_df[col], errors="coerce"
                    )

    for c in feature_list:
        if c not in out.columns:
            out[c] = FEATURE_DEFAULTS.get(c, 0.0)
    # Convert to numeric but preserve NaN — XGBoost handles missing values
    # natively and will learn the optimal split direction.  Replacing NaN
    # with 0.0 is dangerous because 0 carries semantic meaning for many
    # features (e.g. clv=0 means "exactly accurate odds", injury_delta=0
    # means "no injury advantage").
    out[feature_list] = out[feature_list].apply(pd.to_numeric, errors="coerce")

    # Derive sharp_implied_prob from odds when it has near-zero variance
    # (common for raw imported datasets that lack engineered columns).
    # Strip the vig so implied probabilities are not systematically inflated.
    if "sharp_implied_prob" in feature_list and "odds" in out.columns:
        if float(out["sharp_implied_prob"].var()) < EPS:
            valid_odds = out["odds"].where(out["odds"] > 1.0)
            raw_prob = (1.0 / valid_odds).fillna(0.0)
            # Use per-row overround if sharp_vig is available, else 0
            if "sharp_vig" in out.columns:
                overround = 1.0 + out["sharp_vig"]
                overround = overround.where(overround > 0, 1.0)
                out["sharp_implied_prob"] = (raw_prob / overround).clip(0.0, 1.0).fillna(0.0)
            else:
                out["sharp_implied_prob"] = raw_prob.clip(0.0, 1.0)

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
    if "market_momentum" in out.columns:
        out["market_momentum"] = out["market_momentum"].clip(-0.5, 0.5)
    # Phase 4 features
    for col in ("team_attack_strength", "team_defense_strength",
                "opp_attack_strength", "opp_defense_strength"):
        if col in out.columns:
            out[col] = out[col].clip(0.1, 5.0)
    if "expected_total_proxy" in out.columns:
        out["expected_total_proxy"] = out["expected_total_proxy"].clip(0.0, 10.0)
    if "form_trend_slope" in out.columns:
        out["form_trend_slope"] = out["form_trend_slope"].clip(-3.0, 3.0)
    if "rest_fatigue_score" in out.columns:
        out["rest_fatigue_score"] = out["rest_fatigue_score"].clip(0.0, 1.0)
    if "schedule_congestion" in out.columns:
        out["schedule_congestion"] = out["schedule_congestion"].clip(0.0, 1.0)
    for col in ("over25_rate", "btts_rate"):
        if col in out.columns:
            out[col] = out[col].clip(0.0, 1.0)
    if "home_away_split_delta" in out.columns:
        out["home_away_split_delta"] = out["home_away_split_delta"].clip(-1.0, 1.0)
    if "league_position_delta" in out.columns:
        out["league_position_delta"] = out["league_position_delta"].clip(-20.0, 20.0)
    if "goals_scored_avg" in out.columns:
        out["goals_scored_avg"] = out["goals_scored_avg"].clip(0.0, 5.0)
    if "goals_conceded_avg" in out.columns:
        out["goals_conceded_avg"] = out["goals_conceded_avg"].clip(0.0, 5.0)
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
    """Train XGBoost with isotonic calibration and a strict holdout test set.

    The final 20% of the data (chronologically) is held out *before*
    CalibratedClassifierCV sees anything, so the calibrator's isotonic
    regression has never touched the test samples.  This prevents the
    previous data leak where the calibrator's validation folds overlapped
    with the evaluation set.
    """
    X_active = X[active_features].values
    y_vals = y.values

    # --- strict temporal holdout (last 20%) ---
    holdout_frac = 0.20
    split_idx = max(1, int(len(X_active) * (1.0 - holdout_frac)))
    X_train, X_holdout = X_active[:split_idx], X_active[split_idx:]
    y_train, y_holdout = y_vals[:split_idx], y_vals[split_idx:]

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

    # Calibrate with TimeSeriesSplit on *training* data only
    n_splits = min(5, max(2, len(X_train) // 200))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=tscv,
    )
    calibrated.fit(X_train, y_train)

    # Evaluate on the strict holdout that the calibrator never saw
    metrics = _evaluate_holdout(calibrated, X_holdout, y_holdout, y_train)

    return calibrated, metrics


def _evaluate_holdout(
    model: CalibratedClassifierCV,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_train: np.ndarray,
) -> Dict:
    """Evaluate model on a strict holdout set with reliability diagram data.

    Also computes a *dummy* Brier score (predicting the training-set
    class average for every sample) so downstream validation can compare
    the model against a meaningful baseline instead of the arbitrary 0.25
    threshold.
    """
    metrics: Dict = {}
    if len(X_test) == 0:
        return metrics

    y_pred = model.predict_proba(X_test)[:, 1]

    metrics["brier_score"] = round(float(brier_score_loss(y_test, y_pred)), 6)
    metrics["log_loss"] = round(float(log_loss(y_test, y_pred)), 6)
    metrics["test_size"] = len(X_test)

    # Dummy baseline: always predict the training-set win rate
    train_mean = float(y_train.mean()) if len(y_train) > 0 else 0.5
    dummy_pred = np.full(len(y_test), train_mean)
    metrics["brier_score_dummy"] = round(float(brier_score_loss(y_test, dummy_pred)), 6)

    # Reliability diagram: bin predictions into buckets and compute
    # actual vs predicted rates + adjustment factors for Kelly scaling.
    bins = [0.0, 0.3, 0.45, 0.55, 0.7, 1.0]
    reliability_bins: Dict[str, Dict] = {}
    for i in range(len(bins) - 1):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 5:
            actual_rate = float(y_test[mask].mean())
            predicted_rate = float(y_pred[mask].mean())
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
    brier_dummy = metrics.get("brier_score_dummy")
    if brier_dummy is not None and brier >= brier_dummy:
        warnings.append(
            f"Brier score {brier:.4f} >= dummy baseline {brier_dummy:.4f} "
            f"(model is no better than always predicting the training win-rate)."
        )
    elif brier > 0.25:
        warnings.append(f"Brier score {brier:.4f} > 0.25 (worse than balanced-class random). Model may be unreliable.")

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
        query = select(PlacedBet).where(
            PlacedBet.status.in_(["won", "lost"])
        ).order_by(PlacedBet.created_at.asc())
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


def _champion_challenger(
    challenger_metrics: Dict,
    sport_group: str,
) -> Tuple[bool, str]:
    """Compare a challenger model against the currently deployed champion.

    The challenger is promoted ONLY if it achieves a strictly better
    log loss on the holdout set.  This prevents model degradation from
    anomalous training data or overfitting.

    Returns (promoted: bool, reason: str).
    """
    champion_data = load_model(sport_group)
    if champion_data is None:
        return True, "no_existing_champion"

    champ_metrics = champion_data.get("metrics", {})
    champ_ll = champ_metrics.get("log_loss", 999.0)
    champ_brier = champ_metrics.get("brier_score", 999.0)
    chall_ll = challenger_metrics.get("log_loss", 999.0)
    chall_brier = challenger_metrics.get("brier_score", 999.0)

    # Primary gate: log loss must be strictly better
    if chall_ll < champ_ll:
        return True, (
            f"promoted: challenger log_loss={chall_ll:.6f} < champion={champ_ll:.6f} "
            f"(brier: {chall_brier:.6f} vs {champ_brier:.6f})"
        )

    # Secondary: if log loss is within 0.001, check Brier
    if abs(chall_ll - champ_ll) < 0.001 and chall_brier < champ_brier:
        return True, (
            f"promoted (brier tiebreak): {chall_brier:.6f} < {champ_brier:.6f}"
        )

    return False, (
        f"rejected: challenger log_loss={chall_ll:.6f} >= champion={champ_ll:.6f} "
        f"(brier: {chall_brier:.6f} vs {champ_brier:.6f})"
    )


def auto_train_all_models(min_samples: int = 2000) -> str:
    """Train general + sport-specific models with Champion vs Challenger.

    A newly trained model (Challenger) is only promoted to production if
    it achieves a strictly better log loss than the currently deployed
    model (Champion) on the holdout set.  This prevents model degradation.
    """
    MODELS_DIR.mkdir(exist_ok=True)

    with SessionLocal() as db:
        query = select(PlacedBet).where(
            PlacedBet.status.in_(["won", "lost"])
        ).order_by(PlacedBet.created_at.asc())
        df = pd.read_sql(query, db.bind)

    if len(df) < min_samples:
        return f"Zu wenig Daten: {len(df)}/{min_samples}"

    df = _clean_frame(df, FEATURES)
    results = []

    # --- Train general model ---
    X_all = df[FEATURES]
    y_all = (df["status"] == "won").astype(int)
    active_all = _get_active_features(X_all, FEATURES)

    if active_all:
        model, metrics = _train_xgboost(X_all, y_all, active_all)
        warnings = _validate_model(model, metrics, active_all)
        for w in warnings:
            log.warning("ML validation (general): %s", w)

        promoted, reason = _champion_challenger(metrics, "general")
        if promoted:
            joblib.dump({"model": model, "features": active_all, "metrics": metrics, "n_samples": len(df)}, _model_path("general"))
            results.append(f"general: PROMOTED ({len(df)} samples, brier={metrics.get('brier_score', 'N/A')})")
            _save_legacy_weights(df, y_all, active_all, metrics)
        else:
            results.append(f"general: REJECTED — {reason}")
        log.info("Champion/Challenger (general): %s", reason)

    # --- Train sport-specific models ---
    if "sport" in df.columns:
        df["sport_group"] = df["sport"].apply(_get_sport_group)

        sport_extra_map = {
            "soccer": SOCCER_EXTRA_FEATURES,
            "basketball": BASKETBALL_EXTRA_FEATURES,
        }

        for group in ["soccer", "basketball", "tennis", "americanfootball", "icehockey"]:
            subset = df[df["sport_group"] == group]
            if len(subset) < min_samples:
                results.append(f"{group}: skipped ({len(subset)} < {min_samples})")
                continue

            extra = sport_extra_map.get(group, [])
            sport_features = FEATURES + [f for f in extra if f not in FEATURES]

            subset = _clean_frame(subset, sport_features)
            X_sport = subset[sport_features]
            y_sport = (subset["status"] == "won").astype(int)
            active_sport = _get_active_features(X_sport, sport_features)

            if not active_sport:
                results.append(f"{group}: no feature variance")
                continue

            model_sport, metrics_sport = _train_xgboost(X_sport, y_sport, active_sport)
            warnings_sport = _validate_model(model_sport, metrics_sport, active_sport)
            for w in warnings_sport:
                log.warning("ML validation (%s): %s", group, w)

            promoted, reason = _champion_challenger(metrics_sport, group)
            if promoted:
                joblib.dump(
                    {"model": model_sport, "features": active_sport, "metrics": metrics_sport, "n_samples": len(subset)},
                    _model_path(group),
                )
                results.append(f"{group}: PROMOTED ({len(subset)} samples, brier={metrics_sport.get('brier_score', 'N/A')})")
            else:
                results.append(f"{group}: REJECTED — {reason}")
            log.info("Champion/Challenger (%s): %s", group, reason)

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


def should_retrain(threshold: int = 500) -> Tuple[bool, int]:
    """Check whether enough new graded bets have accumulated since the last training.

    Compares the current count of graded bets in the DB against the
    ``n_samples`` stored in the deployed champion model.  Triggers
    retraining when ``threshold`` (default 500) new bets are available.

    Returns (should_retrain, new_bets_count).
    """
    champion = load_model("general")
    last_n = champion.get("n_samples", 0) if champion else 0

    with SessionLocal() as db:
        from sqlalchemy import func
        total = db.execute(
            select(func.count(PlacedBet.id)).where(
                PlacedBet.status.in_(["won", "lost"])
            )
        ).scalar() or 0

    new_bets = total - last_n
    return new_bets >= threshold, new_bets


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
