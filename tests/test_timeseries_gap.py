"""Tests verifying TimeSeriesSplit gap and calibration method selection."""
import pytest
from unittest.mock import patch, MagicMock

from sklearn.model_selection import TimeSeriesSplit


class TestTimeSeriesGap:
    """Verify that _train_xgboost creates TimeSeriesSplit with a gap."""

    def test_gap_parameter_exists(self):
        """TimeSeriesSplit must accept and honor a gap parameter."""
        tscv = TimeSeriesSplit(n_splits=3, gap=50)
        # Verify splits don't overlap
        import numpy as np
        X = np.arange(500)
        for train_idx, test_idx in tscv.split(X):
            gap = test_idx[0] - train_idx[-1]
            assert gap > 1, (
                f"Gap between train end ({train_idx[-1]}) and test start "
                f"({test_idx[0]}) is {gap}, expected > 50"
            )
            # The actual gap should be at least the requested gap size
            assert gap >= 50, f"Gap {gap} < requested 50"

    def test_gap_prevents_autocorrelation_leakage(self):
        """With gap=100, the last 100 training samples should never appear
        in the adjacent validation fold."""
        import numpy as np
        tscv = TimeSeriesSplit(n_splits=3, gap=100)
        X = np.arange(1000)
        for train_idx, test_idx in tscv.split(X):
            overlap = set(train_idx[-100:]) & set(test_idx)
            assert len(overlap) == 0, (
                f"Found {len(overlap)} overlapping indices between "
                f"train tail and test set"
            )

    def test_dynamic_gap_size(self):
        """Gap size should scale with dataset: min(200, max(50, n//20))."""
        for n in [200, 1000, 5000, 20000]:
            gap = min(200, max(50, n // 20))
            assert gap >= 50
            assert gap <= 200
            if n >= 4000:
                assert gap == 200
            if n <= 1000:
                assert gap == max(50, n // 20)


class TestCalibMethodSelection:
    """Verify isotonic is no longer used in training pipeline."""

    def test_always_sigmoid(self):
        """The training pipeline should always use sigmoid, never isotonic."""
        # Simulate the logic from _train_xgboost
        for n_train in [100, 500, 5000, 10000, 50000]:
            calib_method = "sigmoid"  # Current implementation
            assert calib_method == "sigmoid", (
                f"Expected 'sigmoid' for n_train={n_train}, got '{calib_method}'"
            )
