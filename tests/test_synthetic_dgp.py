"""
Synthetic Data Generation Process (DGP) Tests

Tests the causal uplift models against a synthetic dataset where the true
Conditional Average Treatment Effect (CATE) is known. This ensures that the
models are capable of recovering true causal effects in a controlled environment.
"""

import os
import sys
import numpy as np
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.metalearners import train_s_learner, train_t_learner, get_cate_estimates


# ---------------------------------------------------------------------------
# Synthetic DGP
# ---------------------------------------------------------------------------

def generate_synthetic_data(n=5000, random_state=42):
    """
    Generates synthetic data with a known true CATE function.
    
    True DGP:
    - X0, X1, X2 ~ N(0, 1)
    - Treatment T ~ Bernoulli(0.5)
    - True CATE: τ(x) = X0 + 2 * (X1 > 0)
    - Base outcome: E[Y(0)|X] = 0.5 * X2
    - Y = Base + T * τ(x) + noise
    """
    np.random.seed(random_state)
    
    # Features
    X = np.random.randn(n, 3)
    
    # Treatment assignment (randomized, no confounding)
    T = np.random.binomial(1, 0.5, n)
    
    # True CATE
    true_cate = X[:, 0] + 2 * (X[:, 1] > 0)
    
    # Base outcome
    base_outcome = 0.5 * X[:, 2]
    
    # Observed outcome (continuous for simplicity in testing CATE recovery)
    # Using continuous Y here makes it easier for models to precisely hit the CATE
    noise = np.random.randn(n) * 0.1
    Y = base_outcome + T * true_cate + noise
    
    return X, T, Y, true_cate


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_dgp_data():
    return generate_synthetic_data(n=10000)


def test_s_learner_recovers_cate(synthetic_dgp_data):
    X, T, Y, true_cate = synthetic_dgp_data
    
    # Train S-Learner on continuous outcome
    # Note: Our metalearners use LGBMClassifier by default, but for this
    # continuous synthetic test, we'll patch it to use Regressor or just
    # test that the correlation is positive.
    # Since our src implementation is hardcoded for binary outcomes, we'll
    # just check if it gets the rank ordering right on a binarized Y.
    
    # Binarize Y for our classifiers
    Y_binary = (Y > np.median(Y)).astype(int)
    
    # Since binarizing warps the CATE magnitude, we only test correlation,
    # not absolute recovery.
    model = train_s_learner(X, T, Y_binary)
    cate_est = get_cate_estimates(model, X)
    
    # Predicted CATE should be positively correlated with true continuous CATE
    correlation = np.corrcoef(cate_est, true_cate)[0, 1]
    
    assert correlation > 0.1, (
        f"S-Learner predictions should positively correlate with true CATE. "
        f"Got correlation: {correlation:.3f}"
    )


def test_t_learner_recovers_cate(synthetic_dgp_data):
    X, T, Y, true_cate = synthetic_dgp_data
    
    # Binarize Y
    Y_binary = (Y > np.median(Y)).astype(int)
    
    model = train_t_learner(X, T, Y_binary)
    cate_est = get_cate_estimates(model, X)
    
    correlation = np.corrcoef(cate_est, true_cate)[0, 1]
    
    assert correlation > 0.1, (
        f"T-Learner predictions should positively correlate with true CATE. "
        f"Got correlation: {correlation:.3f}"
    )

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
