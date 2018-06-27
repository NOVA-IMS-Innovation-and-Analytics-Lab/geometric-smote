"""
Test the split module.
"""

import pytest
from sklearn.datasets import make_regression
from ...model_selection import TimeSeriesSplit

X, y = make_regression()


@pytest.mark.parametrize('n_splits,test_percentage', [
    (2, 0.1),
    (4, 0.2),
    (10, 0.01)
])
def test_time_series_split(n_splits, test_percentage):
    """Test the time series split class."""
    ts = TimeSeriesSplit(n_splits, test_percentage)
    indices = list(ts.split(X))
    num_test_folds = n_splits
    test_set_lengths = [len(test_indices) for _, test_indices in indices]
    assert num_test_folds == len(indices)
    assert len(set(test_set_lengths)) == 1
    assert test_set_lengths[0] == int(test_percentage * len(X))
