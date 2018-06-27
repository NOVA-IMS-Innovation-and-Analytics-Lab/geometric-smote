"""
Test the somo module.
"""

import numpy as np
import random as r
from sklearn.datasets import make_classification
from ...over_sampling import SOMO


def test_random_state():
    """Test random state initialization."""
    rand_state = r.randint(0,1000)
    somo = SOMO(random_state=rand_state)
    assert somo.random_state == rand_state


def test_output():
    """Test correct output shape of generated data."""
    len_X = r.randint(3,10)
    X, y = make_classification(n_classes=3, class_sep=2,
                               weights=[0.7, 0.3], n_informative=len_X, n_redundant=0, flip_y=0.1,
                               n_features=len_X, n_clusters_per_class=1, n_samples=200, random_state=10)
    somo = SOMO(som_rows=20, som_cols=20)
    X_res, y_res = somo.fit_sample(X, y)
    assert np.shape(X_res)[1] == len_X


def test_output_ratio():
    """Test Correct ratio of generated synthetic data."""
    X, y = make_classification(n_classes=3, class_sep=2,
                               weights=[0.6, 0.2, 0.2], n_informative=4, n_redundant=0, flip_y=0.1,
                               n_features=5, n_clusters_per_class=1, n_samples=200, random_state=None)
    somo = SOMO(ratio = 'auto', som_rows=20, som_cols=20)
    X_res, y_res = somo.fit_sample(X, y)
    count = [len(y_res[y_res == num]) for num in set(y_res)]
    ratio = [(num/max(count)) for num in count]
    assert 0.99 < np.average(ratio) <= 1


def test_handle_negative_values():
    """Test handle negative values."""
    X, y = make_classification(n_classes=3, class_sep=2,
                               weights=(0.6,0.2,0.2), n_informative=4, n_redundant=0, flip_y=0.1,
                               n_features=5, n_clusters_per_class=1, n_samples=200, random_state=None, shift = -10)
    somo = SOMO(ratio='auto', som_rows=20, som_cols=20)
    X_res, y_res = somo.fit_sample(X, y)
    assert X.mean() < 0
    assert X_res.mean() < 0


def test_multiple_classes():
    """Test validate multiple target classes."""
    for num in range(2,10):
        X, y = make_classification(n_classes=num, class_sep=2, n_informative=4, n_redundant=0, flip_y=0.1,
                                   n_features=5, n_clusters_per_class=1, n_samples=200, random_state=None)
        somo = SOMO(ratio='auto', som_rows=20, som_cols=20)
        X_res, y_res = somo.fit_sample(X, y)
        assert set(y) == set(y_res)
