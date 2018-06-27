"""
Test the dbscan_smote module.
"""

import pytest
import numpy as np
from ...over_sampling import DBSCANSMOTE

@pytest.mark.parametrize("RND_SEED", [(42)])
def test_random_seed(RND_SEED):
    dbsc = DBSCANSMOTE(random_state=RND_SEED)
    assert RND_SEED == dbsc.random_state


X = np.array([[1, 1], [0, 0], [0, 1], [0, -1], [-1, 0], 
        [1, 1], [0, 0], [0, 1], [0, -1], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
y_expected = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
ratio_expected = {0: 0, 1: 6}


@pytest.mark.parametrize("X, y, y_expected, ratio_expected", [(X, y, y_expected, ratio_expected)])
def test_correct_y_shape(X, y, y_expected, ratio_expected):
    dbsn = DBSCANSMOTE(eps=0.2, min_samples=2)
    dbsn.fit(X, y)
    X_, y_ = dbsn.fit_sample(X, y)

    assert y_.shape[0] == np.shape(y_expected)[0]
    assert dbsn.ratio_ == ratio_expected


eps_wrong = np.array([-1, -.3, 0])
min_samples = np.array([2, 3, 4, 5, 6])


@pytest.mark.parametrize("eps", [(-1), (-2), (-0.5)])
def test_eps_fails(eps):
    with pytest.raises(ValueError):
        dbsc = DBSCANSMOTE(eps=eps)


@pytest.mark.parametrize("min_samples", [(1), (2), (4), (20)])
def test_min_samples(min_samples):
    dbsc = DBSCANSMOTE(min_samples=min_samples)

    assert dbsc.min_samples == min_samples
