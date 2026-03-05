"""Tests for the Diebold-Mariano statistical test in champion/challenger gating."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.ml_trainer import diebold_mariano_test


class TestDieboldMarianoTest:
    """Tests for diebold_mariano_test()."""

    def test_identical_predictions_give_p_one(self):
        """Identical models should have DM stat ~0 and p-value ~1.0."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n).astype(float)
        preds = np.random.uniform(0.3, 0.7, n)

        dm_stat, p_value = diebold_mariano_test(y_true, preds, preds)
        assert abs(dm_stat) < 1e-10
        assert p_value == 1.0

    def test_clearly_better_model_has_significant_p(self):
        """A much better model should have dm_stat > 0 and p < 0.05."""
        np.random.seed(42)
        n = 500
        y_true = np.random.randint(0, 2, n).astype(float)

        # Good predictions: close to true labels
        pred_good = y_true * 0.8 + (1 - y_true) * 0.2 + np.random.normal(0, 0.05, n)
        pred_good = np.clip(pred_good, 0.05, 0.95)

        # Bad predictions: noisy
        pred_bad = np.full(n, 0.5)

        dm_stat, p_value = diebold_mariano_test(y_true, pred_bad, pred_good)
        assert dm_stat > 0, "DM stat should be positive when challenger is better"
        assert p_value < 0.05, f"p-value should be < 0.05 for clearly better model, got {p_value}"

    def test_clearly_worse_model_has_negative_stat(self):
        """A worse challenger should have dm_stat < 0."""
        np.random.seed(42)
        n = 500
        y_true = np.random.randint(0, 2, n).astype(float)

        pred_good = y_true * 0.8 + (1 - y_true) * 0.2 + np.random.normal(0, 0.05, n)
        pred_good = np.clip(pred_good, 0.05, 0.95)

        pred_bad = np.full(n, 0.5)

        dm_stat, p_value = diebold_mariano_test(y_true, pred_good, pred_bad)
        assert dm_stat < 0, "DM stat should be negative when challenger is worse"

    def test_small_sample_returns_no_significance(self):
        """With < 10 samples, should return dm_stat=0, p_value=1.0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        pred_a = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        pred_b = np.array([0.6, 0.4, 0.5, 0.5, 0.6])

        dm_stat, p_value = diebold_mariano_test(y_true, pred_a, pred_b)
        assert dm_stat == 0.0
        assert p_value == 1.0

    def test_marginal_difference_not_significant(self):
        """Very similar models should NOT produce significant p-value."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n).astype(float)

        # Nearly identical predictions
        preds_a = np.random.uniform(0.3, 0.7, n)
        preds_b = preds_a + np.random.normal(0, 0.001, n)
        preds_b = np.clip(preds_b, 0.01, 0.99)

        dm_stat, p_value = diebold_mariano_test(y_true, preds_a, preds_b)
        assert p_value > 0.10, f"Marginal difference should not be significant, got p={p_value}"
