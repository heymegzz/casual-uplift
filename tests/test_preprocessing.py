"""
Unit tests for src/preprocessing.py

Tests cover:
- Data loading and sampling
- Stratified train/cal/test splitting
- Feature extraction (get_XTY)
- Feature standardization
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import load_data, split_data, get_XTY, standardize_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df():
    """Creates a minimal synthetic DataFrame that mirrors the Criteo schema."""
    np.random.seed(42)
    n = 10_000
    df = pd.DataFrame({
        **{f'f{i}': np.random.randn(n) for i in range(12)},
        'treatment': np.random.binomial(1, 0.85, n),
        'conversion': np.random.binomial(1, 0.003, n),
        'visit': np.random.binomial(1, 0.05, n),
        'exposure': np.ones(n, dtype=int),
    })
    return df


@pytest.fixture
def splits(synthetic_df):
    """Returns (train, cal, test) DataFrames."""
    return split_data(synthetic_df, random_state=42)


# ---------------------------------------------------------------------------
# Tests — split_data
# ---------------------------------------------------------------------------

class TestSplitData:

    def test_split_returns_three_frames(self, splits):
        assert len(splits) == 3, "split_data should return exactly 3 DataFrames"

    def test_split_preserves_total_rows(self, synthetic_df, splits):
        total = sum(len(s) for s in splits)
        assert total == len(synthetic_df), (
            f"Total rows after split ({total}) must equal original ({len(synthetic_df)})"
        )

    def test_split_ratios_approximate(self, synthetic_df, splits):
        n = len(synthetic_df)
        train, cal, test = splits
        assert abs(len(train) / n - 0.70) < 0.02
        assert abs(len(cal) / n - 0.10) < 0.02
        assert abs(len(test) / n - 0.20) < 0.02

    def test_no_row_leakage(self, synthetic_df, splits):
        """Indices across splits must be disjoint."""
        train, cal, test = splits
        assert len(set(train.index) & set(cal.index)) == 0
        assert len(set(train.index) & set(test.index)) == 0
        assert len(set(cal.index) & set(test.index)) == 0

    def test_treatment_rate_preserved(self, synthetic_df, splits):
        """Treatment rate should be approximately maintained across splits."""
        overall_rate = synthetic_df['treatment'].mean()
        for split in splits:
            split_rate = split['treatment'].mean()
            assert abs(split_rate - overall_rate) < 0.03, (
                f"Treatment rate deviated: overall={overall_rate:.3f}, split={split_rate:.3f}"
            )


# ---------------------------------------------------------------------------
# Tests — get_XTY
# ---------------------------------------------------------------------------

class TestGetXTY:

    def test_returns_three_arrays(self, synthetic_df):
        X, T, Y = get_XTY(synthetic_df, outcome='conversion')
        assert isinstance(X, np.ndarray)
        assert isinstance(T, np.ndarray)
        assert isinstance(Y, np.ndarray)

    def test_shapes_consistent(self, synthetic_df):
        X, T, Y = get_XTY(synthetic_df, outcome='conversion')
        assert X.shape[0] == len(synthetic_df)
        assert X.shape[1] == 12  # f0 through f11
        assert T.shape == (len(synthetic_df),)
        assert Y.shape == (len(synthetic_df),)

    def test_treatment_is_binary(self, synthetic_df):
        _, T, _ = get_XTY(synthetic_df, outcome='conversion')
        assert set(np.unique(T)).issubset({0, 1})

    def test_outcome_is_binary(self, synthetic_df):
        _, _, Y = get_XTY(synthetic_df, outcome='conversion')
        assert set(np.unique(Y)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests — standardize_features
# ---------------------------------------------------------------------------

class TestStandardize:

    def test_output_shapes_match_input(self, splits):
        train, cal, test = splits
        X_train, _, _ = get_XTY(train, outcome='conversion')
        X_cal, _, _ = get_XTY(cal, outcome='conversion')
        X_test, _, _ = get_XTY(test, outcome='conversion')

        X_tr_s, X_cal_s, X_te_s, scaler = standardize_features(X_train, X_cal, X_test)
        assert X_tr_s.shape == X_train.shape
        assert X_cal_s.shape == X_cal.shape
        assert X_te_s.shape == X_test.shape

    def test_train_has_zero_mean_unit_var(self, splits):
        train, cal, test = splits
        X_train, _, _ = get_XTY(train, outcome='conversion')
        X_cal, _, _ = get_XTY(cal, outcome='conversion')
        X_test, _, _ = get_XTY(test, outcome='conversion')

        X_tr_s, _, _, _ = standardize_features(X_train, X_cal, X_test)
        np.testing.assert_allclose(X_tr_s.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(X_tr_s.std(axis=0), 1, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
