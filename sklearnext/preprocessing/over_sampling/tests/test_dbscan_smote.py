from sklearnext.preprocessing.over_sampling.dbscan_smote import DBSCANSMOTE
import pytest

import numpy as np


@pytest.mark.parametrize("RND_SEED", [(42)])
def test_random_seed(RND_SEED):
    dbsc = DBSCANSMOTE(random_state=RND_SEED)
    assert RND_SEED == dbsc.random_state
