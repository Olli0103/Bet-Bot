"""EV Diagnostics: per-signal debug payload for transparent EV decomposition.

Writes ``artifacts/ev_diagnostics.jsonl`` with one JSON object per line
per signal evaluated.  Each line contains the full EV breakdown:
raw_prob, calibrated_prob, target_odds, sharp_odds, implied_prob_target,
implied_prob_sharp, vig, tax_rate, EV_final.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
EV_DIAGNOSTICS_FILE = ARTIFACTS_DIR / "ev_diagnostics.jsonl"


def log_ev_diagnostic(
    event_id: str,
    sport: str,
    market: str,
    selection: str,
    raw_prob: float,
    calibrated_prob: float,
    calibration_source: str,
    target_odds: float,
    sharp_odds: float,
    vig: float,
    tax_rate: float,
    ev_final: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Log a single EV diagnostic entry.

    Returns the diagnostic dict for testing/inspection.
    """
    implied_prob_target = 1.0 / target_odds if target_odds > 1.0 else 0.0
    implied_prob_sharp = 1.0 / sharp_odds if sharp_odds > 1.0 else 0.0

    entry: Dict[str, Any] = {
        "event_id": event_id,
        "sport": sport,
        "market": market,
        "selection": selection,
        "raw_prob": round(raw_prob, 6),
        "calibrated_prob": round(calibrated_prob, 6),
        "calibration_source": calibration_source,
        "target_odds": round(target_odds, 4),
        "sharp_odds": round(sharp_odds, 4),
        "implied_prob_target": round(implied_prob_target, 6),
        "implied_prob_sharp": round(implied_prob_sharp, 6),
        "vig": round(vig, 6),
        "tax_rate": round(tax_rate, 4),
        "EV_final": round(ev_final, 6),
    }
    if extra:
        entry.update(extra)

    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(EV_DIAGNOSTICS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.debug("Failed to write EV diagnostic: %s", exc)

    return entry


def log_cycle_calibration_stats(
    avg_raw_prob: float,
    avg_calibrated_prob: float,
    calibration_adjustment_mean: float,
    n_signals: int,
) -> None:
    """Log per-cycle calibration statistics."""
    log.info(
        "Calibration cycle stats: n=%d avg_raw=%.4f avg_cal=%.4f adj_mean=%.4f",
        n_signals,
        avg_raw_prob,
        avg_calibrated_prob,
        calibration_adjustment_mean,
    )
