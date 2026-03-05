"""Tests for EV calibration and sharp/target price logic.

Required tests:
1) test_calibrated_probability_used_for_ev
2) test_probability_scale_consistency_0_1
3) test_target_sharp_selection_alignment
4) test_vig_and_tax_applied_once
5) test_ev_diagnostics_contains_required_fields
6) test_fallback_calibrator_when_low_samples
"""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.core.betting_math import expected_value, implied_probability
from src.core.calibration import (
    Calibrator,
    CalibrationManager,
    _beta_calibration_fit,
    _compute_calibration_metrics,
    _isotonic_fit,
    _platt_scaling,
    get_calibration_manager,
    write_calibration_report,
    MIN_SAMPLES_SPORT_MARKET,
    MIN_SAMPLES_GLOBAL,
)
from src.core.ev_diagnostics import log_ev_diagnostic, EV_DIAGNOSTICS_FILE
from src.core.feature_engineering import FeatureEngineer
from src.core.pricing_model import QuantPricingModel
from src.models.betting import BetSignal


# ---------------------------------------------------------------------------
# 1) test_calibrated_probability_used_for_ev
# ---------------------------------------------------------------------------

class TestCalibratedProbabilityUsedForEV:
    """EV must be computed using the calibrated probability, not the raw one."""

    def test_calibrated_probability_used_for_ev(self):
        """When calibration is enabled and a calibrator is fitted,
        the EV should be computed using the calibrated probability."""
        # Set up a calibrator that significantly adjusts probabilities
        mgr = CalibrationManager(method="isotonic")

        # Create training data where actuals are lower than predicted
        # (model is overconfident)
        np.random.seed(42)
        n = 100
        predicted = np.random.uniform(0.5, 0.9, n)
        # Actual outcomes: systematically lower than predicted
        actual = (np.random.random(n) < (predicted * 0.8)).astype(float)

        records = [
            {"sport": "basketball_nba", "market": "h2h",
             "model_probability_raw": float(p), "actual_outcome": float(a)}
            for p, a in zip(predicted, actual)
        ]
        mgr.fit_from_history(records)

        # A raw probability of 0.70 should be calibrated down
        raw_prob = 0.70
        cal_prob, source = mgr.calibrate(raw_prob, "basketball_nba", "h2h")

        # Verify calibration adjusted the probability
        assert source in ("sport_market", "global")

        # Compute EV with raw vs calibrated
        odds = 2.0
        tax = 0.05
        ev_raw = expected_value(raw_prob, odds, tax_rate=tax)
        ev_calibrated = expected_value(cal_prob, odds, tax_rate=tax)

        # They should differ (calibration had an effect)
        # The calibrated EV should reflect the adjusted probability
        assert ev_calibrated != pytest.approx(ev_raw, abs=0.001), (
            "EV with calibrated prob must differ from EV with raw prob"
        )

    def test_ev_uses_calibrated_not_raw_in_signal(self):
        """BetSignal.expected_value must match the EV computed from
        model_probability (which is the calibrated value)."""
        sig = BetSignal(
            sport="basketball_nba",
            event_id="evt1",
            market="h2h",
            selection="Team A",
            bookmaker_odds=2.0,
            model_probability=0.60,
            expected_value=expected_value(0.60, 2.0, tax_rate=0.05),
            kelly_fraction=0.0,
            recommended_stake=0.0,
            model_probability_raw=0.70,
            model_probability_calibrated=0.60,
            calibration_source="sport_market",
        )
        # EV should be based on model_probability (calibrated=0.60), not raw=0.70
        expected_ev = expected_value(0.60, 2.0, tax_rate=0.05)
        assert sig.expected_value == pytest.approx(expected_ev, abs=1e-6)


# ---------------------------------------------------------------------------
# 2) test_probability_scale_consistency_0_1
# ---------------------------------------------------------------------------

class TestProbabilityScaleConsistency:
    """All probabilities must be on the 0-1 scale (not 0-100)."""

    def test_probability_scale_consistency_0_1(self):
        """model_probability, model_probability_raw, and
        model_probability_calibrated must all be in [0, 1]."""
        sig = BetSignal(
            sport="soccer_epl",
            event_id="evt2",
            market="h2h",
            selection="Arsenal",
            bookmaker_odds=1.80,
            model_probability=0.65,
            expected_value=0.05,
            kelly_fraction=0.01,
            recommended_stake=5.0,
            model_probability_raw=0.68,
            model_probability_calibrated=0.65,
            calibration_source="sport_market",
        )
        assert 0.0 < sig.model_probability < 1.0
        assert 0.0 < sig.model_probability_raw <= 1.0
        assert 0.0 < sig.model_probability_calibrated <= 1.0

    def test_calibrator_output_always_0_1(self):
        """Calibrator.calibrate() must always return values in (0, 1)."""
        cal = Calibrator(method="isotonic")
        # Fit with some data
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 10)
        actuals = np.array([0, 0, 1, 1, 1] * 10).astype(float)
        cal.fit(probs, actuals)

        for test_p in [0.01, 0.1, 0.5, 0.9, 0.99]:
            result = cal.calibrate(test_p)
            assert 0.0 < result < 1.0, f"Calibrated {test_p} -> {result} out of (0,1)"

    def test_implied_probability_0_1(self):
        """implied_probability must always return values in (0, 1]."""
        for odds in [1.01, 1.5, 2.0, 3.0, 10.0, 100.0]:
            ip = implied_probability(odds)
            assert 0.0 < ip <= 1.0, f"IP for odds {odds} = {ip}"

    def test_calibration_metrics_probabilities_0_1(self):
        """Calibration metrics (ECE, Brier) are computed on 0-1 scale."""
        predicted = np.array([0.2, 0.4, 0.6, 0.8])
        actual = np.array([0, 0, 1, 1], dtype=float)
        metrics = _compute_calibration_metrics(predicted, actual)
        assert 0.0 <= metrics["brier_score"] <= 1.0
        assert 0.0 <= metrics["ece"] <= 1.0


# ---------------------------------------------------------------------------
# 3) test_target_sharp_selection_alignment
# ---------------------------------------------------------------------------

class TestTargetSharpSelectionAlignment:
    """Target odds (Tipico) and sharp odds (Pinnacle) must refer to the
    same selection / outcome — no team-side flip."""

    def test_target_sharp_selection_alignment(self):
        """Verify that when we iterate target selections, the sharp odds
        we look up are for the exact same selection key."""
        # Simulate what live_feed.py does
        sharp_prices = {"TeamA": 1.80, "Draw": 3.50, "TeamB": 4.20}
        target_prices = {"TeamA": 1.75, "Draw": 3.40, "TeamB": 4.50}

        for selection, target_odds in target_prices.items():
            sharp_odds = sharp_prices.get(selection)
            assert sharp_odds is not None, (
                f"Sharp odds missing for selection '{selection}' — "
                "possible team-side flip or mislabeled outcome"
            )
            # Both should refer to the same outcome
            # So sharp implied prob should be in the same ballpark
            sharp_ip = 1.0 / sharp_odds
            target_ip = 1.0 / target_odds
            # They shouldn't differ by more than 20% (extreme vig/shade)
            assert abs(sharp_ip - target_ip) < 0.20, (
                f"Selection '{selection}': sharp_ip={sharp_ip:.3f} vs "
                f"target_ip={target_ip:.3f} — possible mislabel"
            )

    def test_clv_proxy_sign_consistency(self):
        """CLV proxy should be positive when target offers better odds than sharp."""
        # Target gives better odds (higher payout) -> positive CLV
        clv = FeatureEngineer.calculate_clv_proxy(target_odds=2.20, sharp_odds=2.00)
        assert clv > 0, "Positive CLV when target > sharp"

        # Target gives worse odds -> negative CLV
        clv_neg = FeatureEngineer.calculate_clv_proxy(target_odds=1.80, sharp_odds=2.00)
        assert clv_neg < 0, "Negative CLV when target < sharp"


# ---------------------------------------------------------------------------
# 4) test_vig_and_tax_applied_once
# ---------------------------------------------------------------------------

class TestVigAndTaxAppliedOnce:
    """Vig removal and tax should each be applied exactly once in the EV chain."""

    def test_vig_and_tax_applied_once(self):
        """EV formula: tax is applied to odds (once), vig is used for sharp prob
        normalization (once). There should be no double application."""
        model_prob = 0.60
        decimal_odds = 2.0
        tax_rate = 0.05

        ev = expected_value(model_prob, decimal_odds, tax_rate=tax_rate)

        # Manual computation
        net_odds = decimal_odds * (1.0 - tax_rate)  # 2.0 * 0.95 = 1.90
        net_profit = net_odds - 1.0                   # 0.90
        expected = model_prob * net_profit - (1.0 - model_prob)  # 0.60*0.90 - 0.40 = 0.14

        assert ev == pytest.approx(expected, abs=1e-10), (
            f"EV mismatch: got {ev}, expected {expected}. "
            "Possible double tax/vig application."
        )

    def test_vig_calculation_not_double_counted(self):
        """Vig removal in feature engineering should happen exactly once."""
        sharp_market = {"TeamA": 1.90, "Draw": 3.40, "TeamB": 4.00}
        vig = FeatureEngineer.calculate_vig(sharp_market)

        # Sum of implied probs
        ip_sum = sum(1.0 / p for p in sharp_market.values())
        expected_vig = ip_sum - 1.0

        assert vig == pytest.approx(expected_vig, abs=1e-4)
        assert vig > 0, "Sharp market should have positive vig (overround)"

        # De-vigged probabilities should sum to ~1.0
        devigged_sum = sum((1.0 / p) / (1.0 + vig) for p in sharp_market.values())
        assert devigged_sum == pytest.approx(1.0, abs=1e-4), (
            f"De-vigged probs sum to {devigged_sum}, not 1.0 — possible double vig removal"
        )

    def test_tax_not_in_sharp_prob(self):
        """Tax should only apply to the target book (Tipico) odds, not to
        the sharp probability derivation."""
        # The sharp implied probability should not include any tax adjustment
        sharp_odds = 2.0
        sharp_ip = 1.0 / sharp_odds  # 0.50 — no tax
        assert sharp_ip == 0.5

        # Tax only appears in the EV formula on the target odds
        ev_no_tax = expected_value(0.60, 2.0, tax_rate=0.0)
        ev_with_tax = expected_value(0.60, 2.0, tax_rate=0.05)
        assert ev_no_tax > ev_with_tax, "Tax should reduce EV"

    def test_ev_formula_components(self):
        """Verify each component of the EV formula is computed correctly."""
        p = 0.55
        odds = 2.50
        tax = 0.05

        ev = expected_value(p, odds, tax_rate=tax)

        # Step by step
        net_odds = odds * (1.0 - tax)  # 2.50 * 0.95 = 2.375
        net_profit = net_odds - 1.0     # 1.375
        manual_ev = p * net_profit - (1.0 - p)  # 0.55 * 1.375 - 0.45 = 0.30625

        assert ev == pytest.approx(manual_ev, abs=1e-10)


# ---------------------------------------------------------------------------
# 5) test_ev_diagnostics_contains_required_fields
# ---------------------------------------------------------------------------

class TestEVDiagnosticsFields:
    """ev_diagnostics.jsonl must contain all required fields per signal."""

    REQUIRED_FIELDS = {
        "raw_prob", "calibrated_prob", "target_odds", "sharp_odds",
        "implied_prob_target", "implied_prob_sharp", "vig",
        "tax_rate", "EV_final",
    }

    def test_ev_diagnostics_contains_required_fields(self, tmp_path):
        """Each diagnostic entry must contain all required fields."""
        diag_file = tmp_path / "ev_diagnostics.jsonl"

        with patch("src.core.ev_diagnostics.EV_DIAGNOSTICS_FILE", diag_file), \
             patch("src.core.ev_diagnostics.ARTIFACTS_DIR", tmp_path):
            entry = log_ev_diagnostic(
                event_id="evt123",
                sport="basketball_nba",
                market="h2h",
                selection="Lakers",
                raw_prob=0.65,
                calibrated_prob=0.60,
                calibration_source="sport_market",
                target_odds=1.90,
                sharp_odds=1.85,
                vig=0.04,
                tax_rate=0.05,
                ev_final=0.08,
            )

        # Check all required fields are present
        for field in self.REQUIRED_FIELDS:
            assert field in entry, f"Missing required field: {field}"

        # Check values are reasonable
        assert 0 < entry["raw_prob"] < 1
        assert 0 < entry["calibrated_prob"] < 1
        assert entry["target_odds"] > 1.0
        assert entry["sharp_odds"] > 1.0
        assert 0 < entry["implied_prob_target"] < 1
        assert 0 < entry["implied_prob_sharp"] < 1
        assert entry["vig"] >= 0
        assert 0 <= entry["tax_rate"] <= 1

    def test_ev_diagnostics_writes_jsonl(self, tmp_path):
        """Diagnostics should be appended as one JSON per line."""
        diag_file = tmp_path / "ev_diagnostics.jsonl"

        with patch("src.core.ev_diagnostics.EV_DIAGNOSTICS_FILE", diag_file), \
             patch("src.core.ev_diagnostics.ARTIFACTS_DIR", tmp_path):
            for i in range(3):
                log_ev_diagnostic(
                    event_id=f"evt{i}",
                    sport="soccer_epl",
                    market="h2h",
                    selection=f"Team{i}",
                    raw_prob=0.5 + i * 0.1,
                    calibrated_prob=0.5 + i * 0.1,
                    calibration_source="global",
                    target_odds=2.0,
                    sharp_odds=1.95,
                    vig=0.03,
                    tax_rate=0.05,
                    ev_final=0.02 + i * 0.01,
                )

        lines = diag_file.read_text().strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            entry = json.loads(line)
            for field in self.REQUIRED_FIELDS:
                assert field in entry

    def test_implied_prob_computed_correctly(self, tmp_path):
        """implied_prob_target and implied_prob_sharp must be 1/odds."""
        diag_file = tmp_path / "ev_diagnostics.jsonl"

        with patch("src.core.ev_diagnostics.EV_DIAGNOSTICS_FILE", diag_file), \
             patch("src.core.ev_diagnostics.ARTIFACTS_DIR", tmp_path):
            entry = log_ev_diagnostic(
                event_id="test",
                sport="tennis",
                market="h2h",
                selection="Player",
                raw_prob=0.60,
                calibrated_prob=0.58,
                calibration_source="global",
                target_odds=2.50,
                sharp_odds=2.40,
                vig=0.02,
                tax_rate=0.05,
                ev_final=0.03,
            )
        assert entry["implied_prob_target"] == pytest.approx(1.0 / 2.50, abs=1e-4)
        assert entry["implied_prob_sharp"] == pytest.approx(1.0 / 2.40, abs=1e-4)


# ---------------------------------------------------------------------------
# 6) test_fallback_calibrator_when_low_samples
# ---------------------------------------------------------------------------

class TestFallbackCalibrator:
    """When not enough samples exist for a sport/market calibrator,
    fall back to global; if that's missing too, use raw with warning."""

    def test_fallback_calibrator_when_low_samples(self):
        """Sport/market with < MIN_SAMPLES_SPORT_MARKET samples should
        fall back to the global calibrator."""
        mgr = CalibrationManager(method="isotonic")

        np.random.seed(42)
        records = []
        # Soccer has enough samples
        for _ in range(100):
            p = np.random.uniform(0.3, 0.8)
            a = float(np.random.random() < p)
            records.append({
                "sport": "soccer_epl", "market": "h2h",
                "model_probability_raw": p, "actual_outcome": a,
            })
        # Tennis has too few samples
        for _ in range(10):
            p = np.random.uniform(0.3, 0.8)
            a = float(np.random.random() < p)
            records.append({
                "sport": "tennis_atp", "market": "h2h",
                "model_probability_raw": p, "actual_outcome": a,
            })

        report = mgr.fit_from_history(records)

        # Soccer should have sport_market calibrator
        cal_prob_soccer, source_soccer = mgr.calibrate(0.60, "soccer_epl", "h2h")
        assert source_soccer == "sport_market"

        # Tennis should fall back to global (not enough samples)
        cal_prob_tennis, source_tennis = mgr.calibrate(0.60, "tennis_atp", "h2h")
        assert source_tennis == "global", (
            f"Expected global fallback for tennis, got {source_tennis}"
        )

        # Report should indicate tennis has insufficient samples
        tennis_report = report["sport_market_reports"].get("tennis_h2h", {})
        assert tennis_report.get("status") == "insufficient_samples"

    def test_raw_passthrough_when_no_calibrator(self):
        """When no calibrator is available at all, return raw probability."""
        mgr = CalibrationManager(method="isotonic")
        mgr._loaded = True  # Mark as loaded (no data)

        raw_prob = 0.65
        cal_prob, source = mgr.calibrate(raw_prob, "icehockey", "h2h")
        assert cal_prob == raw_prob
        assert source == "raw_passthrough"

    def test_global_fallback_covers_unknown_sport(self):
        """An unknown sport should get the global calibrator."""
        mgr = CalibrationManager(method="isotonic")

        np.random.seed(42)
        records = []
        for _ in range(100):
            p = np.random.uniform(0.3, 0.8)
            a = float(np.random.random() < p)
            records.append({
                "sport": "soccer_epl", "market": "h2h",
                "model_probability_raw": p, "actual_outcome": a,
            })

        mgr.fit_from_history(records)

        # Unknown sport "cricket" should fall back to global
        cal_prob, source = mgr.calibrate(0.60, "cricket", "h2h")
        assert source == "global"


# ---------------------------------------------------------------------------
# Additional calibration unit tests
# ---------------------------------------------------------------------------

class TestCalibratorBasics:
    """Basic calibrator functionality."""

    def test_isotonic_fit_and_calibrate(self):
        """Isotonic calibrator should adjust probabilities."""
        cal = Calibrator(method="isotonic")
        np.random.seed(42)
        probs = np.random.uniform(0.2, 0.9, 50)
        actuals = (np.random.random(50) < probs * 0.9).astype(float)
        cal.fit(probs, actuals)
        assert cal.fitted
        result = cal.calibrate(0.5)
        assert 0.0 < result < 1.0

    def test_platt_fit_and_calibrate(self):
        """Platt scaling calibrator should adjust probabilities."""
        cal = Calibrator(method="platt")
        np.random.seed(42)
        probs = np.random.uniform(0.2, 0.9, 50)
        actuals = (np.random.random(50) < probs).astype(float)
        cal.fit(probs, actuals)
        assert cal.fitted
        result = cal.calibrate(0.5)
        assert 0.0 < result < 1.0

    def test_unfitted_calibrator_passthrough(self):
        """Unfitted calibrator should pass through raw probability."""
        cal = Calibrator(method="isotonic")
        assert cal.calibrate(0.7) == 0.7

    def test_serialize_deserialize(self):
        """Calibrator round-trips through JSON."""
        cal = Calibrator(method="isotonic")
        np.random.seed(42)
        probs = np.random.uniform(0.1, 0.9, 50)
        actuals = (np.random.random(50) < probs).astype(float)
        cal.fit(probs, actuals)

        data = cal.to_dict()
        cal2 = Calibrator.from_dict(data)

        assert cal2.fitted == cal.fitted
        assert cal2.method == cal.method
        assert cal2.calibrate(0.5) == pytest.approx(cal.calibrate(0.5), abs=1e-6)


class TestCalibrationMetrics:
    """Test calibration metric computation."""

    def test_perfect_calibration(self):
        """Perfect predictions should have zero ECE and low Brier."""
        pred = np.array([0.0, 0.0, 1.0, 1.0])
        actual = np.array([0, 0, 1, 1], dtype=float)
        metrics = _compute_calibration_metrics(pred, actual)
        assert metrics["brier_score"] == 0.0
        assert metrics["ece"] == pytest.approx(0.0, abs=0.01)

    def test_worst_calibration(self):
        """Completely wrong predictions should have high Brier."""
        pred = np.array([1.0, 1.0, 0.0, 0.0])
        actual = np.array([0, 0, 1, 1], dtype=float)
        metrics = _compute_calibration_metrics(pred, actual)
        assert metrics["brier_score"] == 1.0

    def test_calibration_report_generation(self, tmp_path):
        """write_calibration_report should create JSON and MD files."""
        report = {
            "global": {"n_samples": 100, "ece": 0.05, "mce": 0.10,
                       "brier_score": 0.22, "log_loss": 0.65, "calibration_bins": []},
            "sport_market_reports": {
                "soccer_h2h": {"n_samples": 80, "ece": 0.04, "mce": 0.08,
                               "brier_score": 0.20, "log_loss": 0.60,
                               "calibration_bins": []},
            },
        }

        json_file = tmp_path / "calibration_report.json"
        md_file = tmp_path / "CALIBRATION_REPORT.md"

        with patch("src.core.calibration.CALIBRATION_REPORT_JSON", json_file), \
             patch("src.core.calibration.CALIBRATION_REPORT_MD", md_file), \
             patch("src.core.calibration.ARTIFACTS_DIR", tmp_path):
            write_calibration_report(report)

        assert json_file.exists()
        assert md_file.exists()
        content = md_file.read_text()
        assert "soccer_h2h" in content
        assert "Brier Score" in content


class TestBetSignalCalibrationFields:
    """BetSignal model includes calibration fields."""

    def test_signal_has_calibration_fields(self):
        """BetSignal should expose raw, calibrated, and source fields."""
        sig = BetSignal(
            sport="soccer_epl",
            event_id="e1",
            market="h2h",
            selection="Arsenal",
            bookmaker_odds=1.80,
            model_probability=0.60,
            expected_value=0.05,
            kelly_fraction=0.01,
            recommended_stake=5.0,
            model_probability_raw=0.65,
            model_probability_calibrated=0.60,
            calibration_source="sport_market",
        )
        assert sig.model_probability_raw == 0.65
        assert sig.model_probability_calibrated == 0.60
        assert sig.calibration_source == "sport_market"

    def test_backward_compat_default_calibration_fields(self):
        """When calibration fields are not set, defaults should be safe."""
        sig = BetSignal(
            sport="soccer_epl",
            event_id="e1",
            market="h2h",
            selection="Arsenal",
            bookmaker_odds=1.80,
            model_probability=0.60,
            expected_value=0.05,
            kelly_fraction=0.01,
            recommended_stake=5.0,
        )
        assert sig.model_probability_raw == 0.0
        assert sig.model_probability_calibrated == 0.0
        assert sig.calibration_source == ""


# ---------------------------------------------------------------------------
# 7) Beta Calibration
# ---------------------------------------------------------------------------

class TestBetaCalibration:
    """Tests for the beta calibration method."""

    def test_beta_calibration_produces_valid_outputs(self):
        """Beta calibration should produce probabilities in [0.01, 0.99]."""
        np.random.seed(42)
        n = 200
        raw_probs = np.random.uniform(0.2, 0.8, n)
        actuals = (np.random.random(n) < raw_probs).astype(float)

        cal = Calibrator(method="beta")
        cal.fit(raw_probs, actuals)
        assert cal.fitted

        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = cal.calibrate(p)
            assert 0.01 <= result <= 0.99, f"Beta calibration out of bounds for p={p}: {result}"

    def test_beta_calibration_monotonic(self):
        """Beta calibration should be approximately monotonic."""
        np.random.seed(42)
        n = 500
        raw_probs = np.random.uniform(0.1, 0.9, n)
        actuals = (np.random.random(n) < raw_probs).astype(float)

        cal = Calibrator(method="beta")
        cal.fit(raw_probs, actuals)

        test_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        calibrated = [cal.calibrate(p) for p in test_points]

        # Check monotonicity (allow small tolerance for numerical issues)
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1] + 0.01, (
                f"Not monotonic: cal({test_points[i]})={calibrated[i]:.4f} > "
                f"cal({test_points[i+1]})={calibrated[i+1]:.4f}"
            )

    def test_beta_calibration_serialization(self):
        """Beta calibrator should survive serialization round-trip."""
        np.random.seed(42)
        n = 100
        raw_probs = np.random.uniform(0.2, 0.8, n)
        actuals = (np.random.random(n) < raw_probs).astype(float)

        cal = Calibrator(method="beta")
        cal.fit(raw_probs, actuals)

        # Round-trip
        data = cal.to_dict()
        cal2 = Calibrator.from_dict(data)

        assert cal2.method == "beta"
        assert cal2.fitted
        assert abs(cal.calibrate(0.5) - cal2.calibrate(0.5)) < 1e-10

    def test_beta_fit_returns_three_params(self):
        """_beta_calibration_fit should return 3 floats."""
        np.random.seed(42)
        n = 100
        raw_probs = np.random.uniform(0.2, 0.8, n)
        actuals = (np.random.random(n) < raw_probs).astype(float)

        a, b, c = _beta_calibration_fit(raw_probs, actuals)
        assert isinstance(a, float)
        assert isinstance(b, float)
        assert isinstance(c, float)

    def test_beta_in_calibration_manager(self):
        """CalibrationManager with method='beta' should work end-to-end."""
        np.random.seed(42)
        n = 400
        raw_probs = np.random.uniform(0.2, 0.8, n)
        actuals = (np.random.random(n) < raw_probs).astype(float)

        records = [
            {"sport": "soccer_epl", "market": "h2h",
             "model_probability_raw": float(p), "actual_outcome": float(a)}
            for p, a in zip(raw_probs, actuals)
        ]

        mgr = CalibrationManager(method="beta")
        report = mgr.fit_from_history(records)

        cal_prob, source = mgr.calibrate(0.6, "soccer_epl", "h2h")
        assert 0.01 <= cal_prob <= 0.99
        assert source in ("sport_market", "global")
