#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
from sqlalchemy import text

from src.data.postgres import engine

ROOT = Path(__file__).resolve().parents[1]

BRIER_TARGETS = {
    "general": 0.235,
    "soccer": 0.240,
    "basketball": 0.245,
    "tennis": 0.205,
    "americanfootball": 0.235,
    "icehockey": 0.250,
}

CRITICAL_BY_SPORT = {
    # Transitional: sentiment/injury are excluded for general/soccer until
    # enough live historical variance accumulates.
    "general": ["sharp_implied_prob", "sharp_vig", "form_winrate_l5", "form_games_l5"],
    "soccer": ["sharp_implied_prob", "sharp_vig", "form_winrate_l5", "form_games_l5"],
    "basketball": ["sharp_implied_prob", "sharp_vig", "form_winrate_l5", "form_games_l5"],
    "tennis": ["sharp_implied_prob", "form_winrate_l5", "form_games_l5"],
    "americanfootball": ["sharp_implied_prob", "form_winrate_l5", "form_games_l5"],
    "icehockey": ["sharp_implied_prob", "form_winrate_l5", "form_games_l5"],
}


def check_alembic_head() -> tuple[bool, str]:
    try:
        out = subprocess.check_output(["./.venv/bin/alembic", "current"], cwd=str(ROOT), text=True, stderr=subprocess.STDOUT)
        return "(head)" in out, out.strip().splitlines()[-1]
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    checks = []

    def add(name: str, ok: bool, detail: str):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    ok, detail = check_alembic_head()
    add("alembic_head", ok, detail)

    cov_path = ROOT / "artifacts" / "feature_coverage.json"
    if cov_path.exists():
        coverage = json.loads(cov_path.read_text())
        failures = []
        for sport, critical_features in CRITICAL_BY_SPORT.items():
            report = coverage.get(sport, {})
            for feat in critical_features:
                stats = report.get(feat)
                if not stats:
                    failures.append(f"{sport}.{feat} missing")
                    continue
                if float(stats.get("non_null_rate", 0.0)) < 0.95:
                    failures.append(f"{sport}.{feat} non_null={stats.get('non_null_rate')}")
                if float(stats.get("variance", 0.0)) <= 0.0:
                    failures.append(f"{sport}.{feat} var=0")

        add(
            "critical_feature_coverage_sport_specific",
            len(failures) == 0,
            "OK" if not failures else "; ".join(failures[:12]),
        )
    else:
        add("critical_feature_coverage_sport_specific", False, "artifacts/feature_coverage.json missing")

    with engine.connect() as conn:
        row = conn.execute(text(
            """
            select count(*) total,
                   sum(case when form_games_l5=0 then 1 else 0 end) g0,
                   count(distinct round(form_winrate_l5::numeric, 3)) wr_bins
            from placed_bets
            where status in ('won','lost','void') and odds is not null and odds > 1.01 and selection is not null
            """
        )).fetchone()

    total, g0, wr_bins = map(int, row)
    g0_rate = (g0 / total) if total else 1.0
    add("form_games_l5_zero_rate_lt_0_10", g0_rate < 0.10, f"g0_rate={g0_rate:.4f} ({g0}/{total})")
    add("form_winrate_l5_unique_bins_gt_10", wr_bins > 10, f"unique_bins={wr_bins}")

    for group, target in BRIER_TARGETS.items():
        model_path = ROOT / "models" / f"xgb_{group}.joblib"
        if not model_path.exists():
            add(f"brier_{group}_target", False, "model missing")
            continue
        data = joblib.load(model_path)
        brier = (data.get("metrics") or {}).get("brier_score")
        add(f"brier_{group}_target", (brier is not None and brier <= target), f"brier={brier} target={target}")

    failed = [c for c in checks if not c["ok"]]
    summary = {
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "status": "PASS" if not failed else "FAIL",
    }

    print(json.dumps({"summary": summary, "checks": checks}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
