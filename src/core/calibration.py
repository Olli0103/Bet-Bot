"""Calibration layer: isotonic regression / Platt scaling per sport+market.

Transforms raw model probabilities into calibrated probabilities that
better reflect actual win rates.  Supports:
- Per sport/market calibrators (e.g. basketball_h2h, soccer_h2h)
- Global fallback when per-sport sample count is too low
- Configurable method: "isotonic" (default) or "platt"
- Safety: returns raw probability with warning when no calibrator available

Calibration data is persisted in ``artifacts/calibration_data.json`` and
loaded at startup.  New calibration fits are triggered by the training
pipeline (``ml_trainer.auto_train_all_models``).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("artifacts")
CALIBRATION_DATA_FILE = ARTIFACTS_DIR / "calibration_data.json"
CALIBRATION_REPORT_JSON = ARTIFACTS_DIR / "calibration_report.json"
CALIBRATION_REPORT_MD = Path("CALIBRATION_REPORT.md")

# Minimum samples required before trusting a per-sport/market ISOTONIC
# calibrator.  Isotonic regression fits a step function — with < 300 samples
# it overfits to noise.  Below this threshold, Platt scaling (logistic) is
# used instead as it has only 2 parameters and generalizes better on small N.
MIN_SAMPLES_SPORT_MARKET = 300
# Minimum samples for the global fallback calibrator
MIN_SAMPLES_GLOBAL = 100


class CalibrationBin:
    """A single bin in a calibration histogram."""

    def __init__(self, low: float, high: float, predicted_mean: float,
                 actual_mean: float, count: int):
        self.low = low
        self.high = high
        self.predicted_mean = predicted_mean
        self.actual_mean = actual_mean
        self.count = count


def _platt_scaling(raw_probs: np.ndarray, actuals: np.ndarray) -> Tuple[float, float]:
    """Fit Platt scaling parameters (A, B) via logistic regression.

    Maps raw probability p to calibrated probability via:
        calibrated = 1 / (1 + exp(A * log_odds(p) + B))

    where log_odds(p) = log(p / (1-p)).
    """
    # Clip to avoid log(0)
    eps = 1e-6
    raw_probs = np.clip(raw_probs, eps, 1.0 - eps)
    log_odds = np.log(raw_probs / (1.0 - raw_probs))

    # Simple gradient descent for logistic regression on log-odds
    a, b = 1.0, 0.0
    lr = 0.01
    for _ in range(1000):
        z = a * log_odds + b
        pred = 1.0 / (1.0 + np.exp(-z))
        pred = np.clip(pred, eps, 1.0 - eps)
        grad_a = np.mean((pred - actuals) * log_odds)
        grad_b = np.mean(pred - actuals)
        a -= lr * grad_a
        b -= lr * grad_b

    return float(a), float(b)


def _isotonic_fit(raw_probs: np.ndarray, actuals: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    """Fit isotonic regression via binned averaging with monotonicity enforcement.

    Returns a list of dicts with keys: low, high, predicted_mean, calibrated_value.
    """
    # Sort by predicted probability
    order = np.argsort(raw_probs)
    sorted_probs = raw_probs[order]
    sorted_actuals = actuals[order]

    n = len(sorted_probs)
    bin_size = max(1, n // n_bins)

    bins: List[Dict[str, float]] = []
    for i in range(0, n, bin_size):
        chunk_probs = sorted_probs[i:i + bin_size]
        chunk_actuals = sorted_actuals[i:i + bin_size]
        bins.append({
            "low": float(chunk_probs[0]),
            "high": float(chunk_probs[-1]),
            "predicted_mean": float(np.mean(chunk_probs)),
            "calibrated_value": float(np.mean(chunk_actuals)),
            "count": int(len(chunk_probs)),
        })

    # Enforce monotonicity (pool adjacent violators)
    i = 0
    while i < len(bins) - 1:
        if bins[i]["calibrated_value"] > bins[i + 1]["calibrated_value"]:
            # Pool bins
            total = bins[i]["count"] + bins[i + 1]["count"]
            pooled_val = (
                bins[i]["calibrated_value"] * bins[i]["count"]
                + bins[i + 1]["calibrated_value"] * bins[i + 1]["count"]
            ) / total
            pooled_pred = (
                bins[i]["predicted_mean"] * bins[i]["count"]
                + bins[i + 1]["predicted_mean"] * bins[i + 1]["count"]
            ) / total
            bins[i] = {
                "low": bins[i]["low"],
                "high": bins[i + 1]["high"],
                "predicted_mean": pooled_pred,
                "calibrated_value": pooled_val,
                "count": total,
            }
            bins.pop(i + 1)
            i = max(0, i - 1)
        else:
            i += 1

    return bins


class Calibrator:
    """Calibrates raw model probabilities for a specific sport/market."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.fitted = False
        self.n_samples = 0
        # Isotonic data
        self._iso_bins: List[Dict[str, float]] = []
        # Platt parameters
        self._platt_a: float = 1.0
        self._platt_b: float = 0.0

    def fit(self, raw_probs: np.ndarray, actuals: np.ndarray) -> None:
        """Fit the calibrator on historical data."""
        if len(raw_probs) < 5:
            log.warning("Calibrator fit skipped: only %d samples", len(raw_probs))
            return

        self.n_samples = len(raw_probs)

        if self.method == "platt":
            self._platt_a, self._platt_b = _platt_scaling(raw_probs, actuals)
        else:
            self._iso_bins = _isotonic_fit(raw_probs, actuals)

        self.fitted = True
        log.info("Calibrator fitted (%s): %d samples", self.method, self.n_samples)

    def calibrate(self, raw_prob: float) -> float:
        """Transform a raw probability into a calibrated one."""
        if not self.fitted:
            return raw_prob

        raw_prob = max(0.01, min(0.99, raw_prob))

        if self.method == "platt":
            eps = 1e-6
            p = max(eps, min(1.0 - eps, raw_prob))
            log_odds = math.log(p / (1.0 - p))
            z = self._platt_a * log_odds + self._platt_b
            return max(0.01, min(0.99, 1.0 / (1.0 + math.exp(-z))))

        # Isotonic: interpolate between bins
        if not self._iso_bins:
            return raw_prob

        # Below first bin
        if raw_prob <= self._iso_bins[0]["predicted_mean"]:
            return max(0.01, min(0.99, self._iso_bins[0]["calibrated_value"]))

        # Above last bin
        if raw_prob >= self._iso_bins[-1]["predicted_mean"]:
            return max(0.01, min(0.99, self._iso_bins[-1]["calibrated_value"]))

        # Linear interpolation between bins
        for i in range(len(self._iso_bins) - 1):
            lo = self._iso_bins[i]
            hi = self._iso_bins[i + 1]
            if lo["predicted_mean"] <= raw_prob <= hi["predicted_mean"]:
                span = hi["predicted_mean"] - lo["predicted_mean"]
                if span < 1e-9:
                    return max(0.01, min(0.99, lo["calibrated_value"]))
                t = (raw_prob - lo["predicted_mean"]) / span
                val = lo["calibrated_value"] + t * (hi["calibrated_value"] - lo["calibrated_value"])
                return max(0.01, min(0.99, val))

        return raw_prob

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON persistence."""
        return {
            "method": self.method,
            "fitted": self.fitted,
            "n_samples": self.n_samples,
            "iso_bins": self._iso_bins,
            "platt_a": self._platt_a,
            "platt_b": self._platt_b,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Calibrator":
        """Deserialize from JSON."""
        cal = cls(method=data.get("method", "isotonic"))
        cal.fitted = data.get("fitted", False)
        cal.n_samples = data.get("n_samples", 0)
        cal._iso_bins = data.get("iso_bins", [])
        cal._platt_a = data.get("platt_a", 1.0)
        cal._platt_b = data.get("platt_b", 0.0)
        return cal


class CalibrationManager:
    """Manages per-sport/market calibrators with global fallback."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self._calibrators: Dict[str, Calibrator] = {}
        self._global_calibrator: Optional[Calibrator] = None
        self._loaded = False

    def _key(self, sport: str, market: str) -> str:
        """Build lookup key from sport and market."""
        s = sport.lower()
        # Normalize sport prefix
        if s.startswith(("soccer", "football")):
            s = "soccer"
        elif s.startswith("basketball"):
            s = "basketball"
        elif s.startswith("tennis"):
            s = "tennis"
        elif s.startswith("icehockey"):
            s = "icehockey"
        elif s.startswith("americanfootball"):
            s = "americanfootball"
        else:
            s = "general"
        return f"{s}_{market.lower()}"

    def fit_from_history(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fit calibrators from historical bet records.

        Each record must have: sport, market, model_probability_raw, actual_outcome (0 or 1).

        Returns a calibration report dict.
        """
        from collections import defaultdict

        # Group by sport/market
        groups: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
        all_probs: List[float] = []
        all_actuals: List[float] = []

        for rec in records:
            sport = rec.get("sport", "general")
            market = rec.get("market", "h2h")
            raw_p = float(rec.get("model_probability_raw", rec.get("model_probability", 0.5)))
            actual = float(rec.get("actual_outcome", 0))
            key = self._key(sport, market)
            groups[key][0].append(raw_p)
            groups[key][1].append(actual)
            all_probs.append(raw_p)
            all_actuals.append(actual)

        report: Dict[str, Any] = {"sport_market_reports": {}, "global": {}}

        # Fit per sport/market
        # - >= MIN_SAMPLES_SPORT_MARKET (300): use configured method (isotonic/platt)
        # - 30-299 samples: force Platt scaling (2 params, safe on small N)
        # - < 30 samples: insufficient, use global fallback
        _PLATT_FALLBACK_MIN = 30
        for key, (probs, actuals) in groups.items():
            n = len(probs)
            if n >= MIN_SAMPLES_SPORT_MARKET:
                cal = Calibrator(method=self.method)
                cal.fit(np.array(probs), np.array(actuals))
                self._calibrators[key] = cal

                metrics = _compute_calibration_metrics(np.array(probs), np.array(actuals))
                report["sport_market_reports"][key] = {
                    "n_samples": n,
                    "method": self.method,
                    **metrics,
                }
            elif n >= _PLATT_FALLBACK_MIN:
                # Too few for isotonic, but Platt scaling (logistic, 2 params) is safe
                cal = Calibrator(method="platt")
                cal.fit(np.array(probs), np.array(actuals))
                self._calibrators[key] = cal

                metrics = _compute_calibration_metrics(np.array(probs), np.array(actuals))
                report["sport_market_reports"][key] = {
                    "n_samples": n,
                    "method": "platt_fallback",
                    "note": f"< {MIN_SAMPLES_SPORT_MARKET} samples, forced Platt scaling",
                    **metrics,
                }
                log.info(
                    "Calibration %s: %d samples < %d, using Platt fallback",
                    key, n, MIN_SAMPLES_SPORT_MARKET,
                )
            else:
                report["sport_market_reports"][key] = {
                    "n_samples": n,
                    "status": "insufficient_samples",
                    "min_required": _PLATT_FALLBACK_MIN,
                }

        # Fit global fallback
        if len(all_probs) >= MIN_SAMPLES_GLOBAL:
            self._global_calibrator = Calibrator(method=self.method)
            self._global_calibrator.fit(np.array(all_probs), np.array(all_actuals))

            metrics = _compute_calibration_metrics(np.array(all_probs), np.array(all_actuals))
            report["global"] = {
                "n_samples": len(all_probs),
                "method": self.method,
                **metrics,
            }

        self._loaded = True
        return report

    def calibrate(
        self,
        raw_prob: float,
        sport: str,
        market: str = "h2h",
    ) -> Tuple[float, str]:
        """Calibrate a raw probability. Returns (calibrated_prob, source).

        source is one of: "sport_market", "global", "raw_passthrough".
        """
        if not self._loaded:
            self._try_load()

        key = self._key(sport, market)

        # Try sport/market calibrator
        cal = self._calibrators.get(key)
        if cal and cal.fitted and cal.n_samples >= MIN_SAMPLES_SPORT_MARKET:
            return cal.calibrate(raw_prob), "sport_market"

        # Fallback to global
        if self._global_calibrator and self._global_calibrator.fitted:
            return self._global_calibrator.calibrate(raw_prob), "global"

        # No calibrator: raw passthrough with warning
        log.debug(
            "No calibrator for %s (sport=%s, market=%s) — using raw probability",
            key, sport, market,
        )
        return raw_prob, "raw_passthrough"

    def save(self) -> None:
        """Persist calibration data to disk."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        data: Dict[str, Any] = {
            "method": self.method,
            "calibrators": {k: v.to_dict() for k, v in self._calibrators.items()},
        }
        if self._global_calibrator:
            data["global_calibrator"] = self._global_calibrator.to_dict()
        with open(CALIBRATION_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        log.info("Calibration data saved to %s", CALIBRATION_DATA_FILE)

    def _try_load(self) -> None:
        """Load calibration data from disk if available."""
        if not CALIBRATION_DATA_FILE.exists():
            self._loaded = True
            return
        try:
            with open(CALIBRATION_DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.method = data.get("method", "isotonic")
            for key, cal_data in data.get("calibrators", {}).items():
                self._calibrators[key] = Calibrator.from_dict(cal_data)
            if "global_calibrator" in data:
                self._global_calibrator = Calibrator.from_dict(data["global_calibrator"])
            self._loaded = True
            log.info("Calibration data loaded: %d sport/market calibrators",
                     len(self._calibrators))
        except Exception as exc:
            log.warning("Failed to load calibration data: %s", exc)
            self._loaded = True


def _compute_calibration_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute ECE, MCE, Brier score, log-loss, and per-bin calibration data."""
    n = len(predicted)
    if n == 0:
        return {}

    # Brier score
    brier = float(np.mean((predicted - actual) ** 2))

    # Log-loss
    eps = 1e-15
    clipped = np.clip(predicted, eps, 1.0 - eps)
    logloss = float(-np.mean(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)))

    # Binned calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins: List[Dict[str, Any]] = []
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (predicted == bin_edges[i + 1])

        bin_count = int(mask.sum())
        if bin_count == 0:
            bins.append({
                "range": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
                "predicted_mean": 0.0,
                "actual_mean": 0.0,
                "count": 0,
            })
            continue

        pred_mean = float(np.mean(predicted[mask]))
        actual_mean = float(np.mean(actual[mask]))
        gap = abs(pred_mean - actual_mean)

        ece += gap * (bin_count / n)
        mce = max(mce, gap)

        bins.append({
            "range": f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}",
            "predicted_mean": round(pred_mean, 4),
            "actual_mean": round(actual_mean, 4),
            "count": bin_count,
            "gap": round(gap, 4),
        })

    return {
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(logloss, 4),
        "calibration_bins": bins,
    }


def write_calibration_report(report: Dict[str, Any]) -> None:
    """Write calibration report to JSON and markdown."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(CALIBRATION_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown
    lines = [
        "# Calibration Report",
        "",
        "## Global Metrics",
        "",
    ]

    global_report = report.get("global", {})
    if global_report:
        lines.append(f"- **Samples:** {global_report.get('n_samples', 0)}")
        lines.append(f"- **ECE:** {global_report.get('ece', 'N/A')}")
        lines.append(f"- **MCE:** {global_report.get('mce', 'N/A')}")
        lines.append(f"- **Brier Score:** {global_report.get('brier_score', 'N/A')}")
        lines.append(f"- **Log Loss:** {global_report.get('log_loss', 'N/A')}")
    else:
        lines.append("No global calibration data available.")

    lines.append("")
    lines.append("## Per Sport/Market")
    lines.append("")

    for key, data in report.get("sport_market_reports", {}).items():
        lines.append(f"### {key}")
        lines.append("")
        if data.get("status") == "insufficient_samples":
            lines.append(
                f"Insufficient samples ({data.get('n_samples', 0)}"
                f" < {data.get('min_required', MIN_SAMPLES_SPORT_MARKET)}). "
                "Using global fallback."
            )
        else:
            lines.append(f"- **Samples:** {data.get('n_samples', 0)}")
            lines.append(f"- **ECE:** {data.get('ece', 'N/A')}")
            lines.append(f"- **MCE:** {data.get('mce', 'N/A')}")
            lines.append(f"- **Brier Score:** {data.get('brier_score', 'N/A')}")
            lines.append(f"- **Log Loss:** {data.get('log_loss', 'N/A')}")

            cal_bins = data.get("calibration_bins", [])
            if cal_bins:
                lines.append("")
                lines.append("| Bin | Predicted | Actual | Count | Gap |")
                lines.append("|-----|-----------|--------|-------|-----|")
                for b in cal_bins:
                    lines.append(
                        f"| {b['range']} | {b.get('predicted_mean', 0):.4f} "
                        f"| {b.get('actual_mean', 0):.4f} "
                        f"| {b.get('count', 0)} "
                        f"| {b.get('gap', 0):.4f} |"
                    )
        lines.append("")

    with open(CALIBRATION_REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Calibration report written: %s, %s", CALIBRATION_REPORT_JSON, CALIBRATION_REPORT_MD)


# Module-level singleton
_manager: Optional[CalibrationManager] = None


def get_calibration_manager(method: str = "isotonic") -> CalibrationManager:
    """Get or create the global CalibrationManager singleton."""
    global _manager
    if _manager is None:
        _manager = CalibrationManager(method=method)
    return _manager
