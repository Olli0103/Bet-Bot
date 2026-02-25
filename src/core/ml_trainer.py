import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select

from src.data.models import PlacedBet
from src.data.postgres import SessionLocal


FEATURES = [
    "sentiment_delta",
    "injury_delta",
    "sharp_implied_prob",
    "clv",
    "sharp_vig",
    "form_winrate_l5",
    "form_games_l5",
]
EPS = 1e-12


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURES:
        if c not in out.columns:
            out[c] = 0.0
    out[FEATURES] = out[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # clamp outliers / enforce sane ranges
    out["sentiment_delta"] = out["sentiment_delta"].clip(-5.0, 5.0)
    out["injury_delta"] = out["injury_delta"].clip(-20.0, 20.0)
    out["sharp_implied_prob"] = out["sharp_implied_prob"].clip(0.0, 1.0)
    out["clv"] = out["clv"].clip(-5.0, 5.0)
    out["sharp_vig"] = out["sharp_vig"].clip(-0.2, 0.5)
    out["form_winrate_l5"] = out["form_winrate_l5"].clip(0.0, 1.0)
    out["form_games_l5"] = out["form_games_l5"].clip(0.0, 5.0)
    return out


def auto_train_model(min_samples: int = 500):
    with SessionLocal() as db:
        query = select(PlacedBet).where(PlacedBet.status.in_(["won", "lost"]))
        df = pd.read_sql(query, db.bind)

    if len(df) < min_samples:
        return f"Zu wenig Daten: {len(df)}/{min_samples}"

    df = _clean_frame(df)
    X = df[FEATURES]
    y = (df["status"] == "won").astype(int)

    variances = X.var(axis=0).to_dict()
    active_features = [f for f in FEATURES if float(variances.get(f, 0.0)) > EPS]
    if not active_features:
        return "Abbruch: Keine Feature-Varianz verfügbar."

    X_active = X[active_features]

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="liblinear")),
        ]
    )
    pipe.fit(X_active, y)

    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]

    coef_std = clf.coef_[0]
    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coef_raw_active = coef_std / scale
    intercept_raw = float(clf.intercept_[0] - np.sum((scaler.mean_ / scale) * coef_std))

    coef_raw_map = {f: 0.0 for f in FEATURES}
    for i, f in enumerate(active_features):
        coef_raw_map[f] = float(coef_raw_active[i])

    learned_weights = {
        "sentiment_delta": float(coef_raw_map["sentiment_delta"]),
        "injury_delta": float(coef_raw_map["injury_delta"]),
        "sharp_implied_prob": float(coef_raw_map["sharp_implied_prob"]),
        "clv": float(coef_raw_map["clv"]),
        "sharp_vig": float(coef_raw_map["sharp_vig"]),
        "form_winrate_l5": float(coef_raw_map["form_winrate_l5"]),
        "form_games_l5": float(coef_raw_map["form_games_l5"]),
        "intercept": intercept_raw,
        "meta": {
            "samples": int(len(df)),
            "wins": int(y.sum()),
            "losses": int((1 - y).sum()),
            "active_features": active_features,
            "variance": {k: float(v) for k, v in variances.items()},
        },
    }

    with open("ml_strategy_weights.json", "w", encoding="utf-8") as f:
        json.dump(learned_weights, f)

    return f"Erfolg: {len(df)} Wetten trainiert. Aktiv: {', '.join(active_features)}"
