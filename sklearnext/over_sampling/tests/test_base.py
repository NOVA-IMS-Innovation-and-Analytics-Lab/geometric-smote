"""
Test the base module.
"""

from itertools import product
from collections import OrderedDict, Counter
from unittest import mock

import pytest
import numpy as np
from sklearn.cluster import KMeans

from ...over_sampling import RandomOverSampler, SMOTE, GeometricSMOTE
from ...cluster import SOM
from ...utils.validation import _TrivialOversampler
from ..distribution import DensityDistributor

X = np.array(list(product(range(5), range(4))))
y = np.array([0] * 10 + [1] * 6 + [2] * 4)
LABELS = np.array([0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 0, 3, 3, 3, 0, 3, 3, 3])
NEIGHBORS = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]


@pytest.mark.parametrize('categorical_cols', [[], (), {}, 'value'])
def test_validate_categorical_cols_type(categorical_cols):
    """Test the type validation of categorical columns for the extended base oversampler."""
    oversampler = _TrivialOversampler(categorical_cols=categorical_cols)
    with pytest.raises(TypeError):    
        oversampler.fit(X, y)


@pytest.mark.parametrize('clusterer', [None, KMeans(), SOM()])
def test_fit(clusterer):
    """Test the fit method of the extended base oversampler."""
    oversampler = _TrivialOversampler(clusterer=clusterer).fit(X, y)
    assert oversampler.ratio_ == {0: 0, 1: 4, 2: 6}
    assert hasattr(oversampler, 'distributor_')
    assert hasattr(oversampler.distributor_, 'intra_distribution_')
    assert hasattr(oversampler.distributor_, 'inter_distribution_')
    if isinstance(clusterer, SOM):
        assert len(oversampler.distributor_.inter_distribution_) > 0
    else:
        assert len(oversampler.distributor_.inter_distribution_) == 0


@pytest.mark.parametrize('X,y,oversampler_class', [
    (np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]), np.array([0, 0, 1, 1, 1]), RandomOverSampler),
    (np.array([(0, 0), (2, 2), (3, 3), (4, 4)]), np.array([0, 1, 1, 1]), RandomOverSampler),
    (np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]), np.array([0, 0, 1, 1, 1]), SMOTE),
    (np.array([(0, 0), (2, 2), (3, 3), (4, 4)]), np.array([0, 1, 1, 1]), SMOTE),
    (np.array([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]), np.array([0, 0, 1, 1, 1]), GeometricSMOTE),
    (np.array([(0, 0), (2, 2), (3, 3), (4, 4)]), np.array([0, 1, 1, 1]), GeometricSMOTE)
])
def test_intra_sample_corner_cases(X, y, oversampler_class):
    """Test the _intra_sample method of the extended base oversampler
    for various corner cases and oversamplers."""
    oversampler = oversampler_class().fit(X, y)
    X_new, y_new = oversampler._intra_sample(X, y, oversampler.ratio_)
    y_count = Counter(y)
    assert X_new.shape == (y_count[1] - y_count[0], X.shape[1])


def test_intra_sample():
    """Test the _intra_sample method of the extended base oversampler."""
    oversampler = SMOTE(clusterer=KMeans(), distributor=DensityDistributor(filtering_threshold=3.0, distances_exponent=0, sparsity_based=False))
    oversampler.fit(X, y)
    initial_ratio = oversampler.ratio_
    k_neighbors = oversampler.k_neighbors
    X_new, y_new = oversampler._intra_sample(X, y, oversampler.ratio_)
    assert oversampler.ratio_ == initial_ratio
    assert oversampler.k_neighbors == k_neighbors


def test_inter_sample():
    """Test the _inter_sample method of the extended base oversampler."""
    oversampler = SMOTE(clusterer=KMeans(), distributor=DensityDistributor(filtering_threshold=3.0, distances_exponent=0, sparsity_based=False))
    oversampler.fit(X, y,)
    initial_ratio = oversampler.ratio_
    k_neighbors = oversampler.k_neighbors
    X_new, y_new = oversampler._inter_sample(X, y, initial_ratio)
    assert oversampler.ratio_ == initial_ratio
    assert oversampler.k_neighbors == k_neighbors


    
