from sklearnext.preprocessing.over_sampling.dbscan_smote import DBSCANSMOTE
import pytest

import numpy as np


@pytest.mark.parametrize("RND_SEED", [(42)])
def test_random_seed(RND_SEED):
    dbsc = DBSCANSMOTE(random_state=RND_SEED)
    assert RND_SEED == dbsc.random_state


X = np.array([[1,1], [0,0], [0,1], [0, -1], [-1, 0], [1,1], [0, 0], [0, 1], [0, -1], [-1, 0], [-1, -1], [1, 1], [0, -1], [-1, 0], [-1, -1], [1, 1]])
y = np.array([0, 0, 0, 0, 0,0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
y_expected = ([0, 0, 0, 0, 0,0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])

@pytest.mark.parametrize("X, y, y_expected", [(X, y, y_expected)])
def test_correct_y_shape(X, y, y_expected):
    dbsn = DBSCANSMOTE()
    X_, y_ = dbsn.fit_sample(X, y)

    assert y_.shape[0] == y_expected.shape[0]
