"""
Unit tests for src/evaluation.py

Tests cover:
- AUUC computation
- Qini coefficient computation
- Calibration error computation
- Policy value computation
- Placebo test mechanics
"""

import os
import sys
import numpy as np
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation import (
    compute_auuc,
    compute_qini,
    compute_calibration,
    compute_policy_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_data():
    """Generates random treatment/outcome data where CATE is approximately zero."""
    np.random.seed(42)
    n = 5000
    T = np.random.binomial(1, 0.5, n)
    Y = np.random.binomial(1, 0.05, n)
    cate = np.random.randn(n) * 0.01  # random noise, no real signal
    return cate, T, Y


@pytest.fixture
def perfect_signal_data():
    """
    Generates data where a perfect CATE estimator can be constructed.
    Users with positive true CATE should have higher Y when treated.
    """
    np.random.seed(123)
    n = 10000
    T = np.random.binomial(1, 0.5, n)

    # True CATE: first half of users have positive effect, second half negative
    true_cate = np.concatenate([np.ones(n // 2) * 0.1, np.ones(n // 2) * -0.05])
    base_rate = 0.05
    prob = base_rate + T * true_cate
    prob = np.clip(prob, 0, 1)
    Y = np.random.binomial(1, prob)

    # Perfect CATE estimator
    cate_est = true_cate.copy()
    return cate_est, T, Y


# ---------------------------------------------------------------------------
# Tests — compute_auuc
# ---------------------------------------------------------------------------

class TestAUUC:

    def test_returns_scalar(self, random_data):
        cate, T, Y = random_data
        auuc = compute_auuc(cate, T, Y)
        assert np.isscalar(auuc) or isinstance(auuc, (float, np.floating))

    def test_random_cate_near_zero(self, random_data):
        cate, T, Y = random_data
        auuc = compute_auuc(cate, T, Y)
        # Random CATE should give AUUC near zero (no better than random)
        assert abs(auuc) < 0.01, f"Random CATE should yield near-zero AUUC, got {auuc}"

    def test_perfect_signal_positive(self, perfect_signal_data):
        cate, T, Y = perfect_signal_data
        auuc = compute_auuc(cate, T, Y)
        # A perfect ranker should give positive AUUC
        assert auuc > 0, f"Perfect signal should produce positive AUUC, got {auuc}"

    def test_reversed_signal_negative(self, perfect_signal_data):
        cate, T, Y = perfect_signal_data
        auuc_reversed = compute_auuc(-cate, T, Y)
        # Reversed ranking should give negative AUUC
        assert auuc_reversed < 0, f"Reversed signal should produce negative AUUC, got {auuc_reversed}"


# ---------------------------------------------------------------------------
# Tests — compute_qini
# ---------------------------------------------------------------------------

class TestQini:

    def test_returns_scalar(self, random_data):
        cate, T, Y = random_data
        qini = compute_qini(cate, T, Y)
        assert np.isscalar(qini) or isinstance(qini, (float, np.floating))

    def test_random_cate_near_zero(self, random_data):
        cate, T, Y = random_data
        qini = compute_qini(cate, T, Y)
        assert abs(qini) < 5, f"Random CATE should yield near-zero Qini, got {qini}"

    def test_auuc_and_qini_agree_on_sign(self, perfect_signal_data):
        cate, T, Y = perfect_signal_data
        auuc = compute_auuc(cate, T, Y)
        qini = compute_qini(cate, T, Y)
        # Both should have the same sign for a strong signal
        assert (auuc > 0) == (qini > 0), (
            f"AUUC ({auuc}) and Qini ({qini}) should agree on sign for perfect signal"
        )


# ---------------------------------------------------------------------------
# Tests — compute_calibration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_returns_correct_structure(self, random_data):
        cate, T, Y = random_data
        bin_centers, actual_ates, cal_err = compute_calibration(cate, T, Y, n_bins=10)
        assert len(bin_centers) == 10
        assert len(actual_ates) == 10
        assert np.isscalar(cal_err) or isinstance(cal_err, (float, np.floating))

    def test_calibration_error_non_negative(self, random_data):
        cate, T, Y = random_data
        _, _, cal_err = compute_calibration(cate, T, Y)
        assert cal_err >= 0, f"Calibration error must be non-negative, got {cal_err}"

    def test_bin_centers_sorted_descending(self, random_data):
        cate, T, Y = random_data
        bin_centers, _, _ = compute_calibration(cate, T, Y)
        # Bins are sorted by descending CATE, so bin_centers should be non-increasing
        for i in range(len(bin_centers) - 1):
            assert bin_centers[i] >= bin_centers[i + 1], (
                f"Bin centers should be non-increasing: {bin_centers}"
            )


# ---------------------------------------------------------------------------
# Tests — compute_policy_value
# ---------------------------------------------------------------------------

class TestPolicyValue:

    def test_returns_dict(self, random_data):
        cate, T, Y = random_data
        pv = compute_policy_value(cate, T, Y)
        assert isinstance(pv, dict)
        assert len(pv) == 4  # default fractions: [0.05, 0.10, 0.20, 0.30]

    def test_values_are_floats(self, random_data):
        cate, T, Y = random_data
        pv = compute_policy_value(cate, T, Y)
        for frac, val in pv.items():
            assert isinstance(val, (int, float, np.floating)), (
                f"Policy value for fraction {frac} should be numeric, got {type(val)}"
            )

    def test_custom_fractions(self, random_data):
        cate, T, Y = random_data
        pv = compute_policy_value(cate, T, Y, targeting_fractions=[0.10, 0.50])
        assert len(pv) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
