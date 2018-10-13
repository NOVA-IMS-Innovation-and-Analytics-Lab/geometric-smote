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
from ..base import (
    _count_clusters_samples,
    _calculate_clusters_density,
    _intra_distribute,
    _inter_distribute,
    density_distribution,
    ExtendedBaseOverSampler
)

X = np.array(list(product(range(5), range(4))))
y = np.array([0] * 10 + [1] * 6 + [2] * 4)
LABELS = np.array([0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 0, 3, 3, 3, 0, 3, 3, 3])
NEIGHBORS = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]


@pytest.mark.parametrize('labels,samples_counts', [
    ([(0, True), (0, True), (0, False), (1, False), (1, False)], [(0, [(True, 2), (False, 1)]), (1, [(False, 2)])]),
    ([(0, True), (2, True), (1, False), (2, False), (1, False)], [(0, [(True, 1)]), (1, [(False, 2)]), (2, [(True, 1), (False, 1)])])
])
def test_count_clusters_samples(labels, samples_counts):
    """Test the count of minority and majority samples in each cluster."""
    _count_clusters_samples(labels) == OrderedDict(samples_counts)


@pytest.mark.parametrize('filtering_threshold,distances_exponent', [
    (1.0, 0),
    (1.0, 1),
    (2.0, 0),
    (2.0, 1)
])
def test_calculate_clusters_density(filtering_threshold, distances_exponent):
    """Test the calculation of filtered clusters density."""
    clustering_labels = np.array([0, 1, 1, 2, 2, 2, 2, 2])
    X = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [2, 0], [3, 0], [4, 0]])
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    clusters_density = _calculate_clusters_density(clustering_labels, X, y, filtering_threshold, distances_exponent)
    if filtering_threshold == 1.0:
        assert clusters_density == {}
    elif filtering_threshold == 2.0 and distances_exponent == 0:
        assert clusters_density == {1: 2.0, 2: 2.0}
    elif filtering_threshold == 1.0 and distances_exponent == 1:
        assert clusters_density == {1: 1.0, 2: 1.0}


@pytest.mark.parametrize('clusters_density,sparsity_based', [
    ({0: 5.0, 1: 15.0}, True),
    ({0: 5.0, 1: 15.0}, False)
])
def test_intra_distribute(clusters_density, sparsity_based):
    """Test the distribution of generated samples in each cluster."""
    distribution = dict(_intra_distribute(clusters_density, sparsity_based, 1.0))
    if sparsity_based:
        np.testing.assert_approx_equal(distribution[0], 0.75)
        np.testing.assert_approx_equal(distribution[1], 0.25)
    else:
        np.testing.assert_approx_equal(distribution[0], 0.25)
        np.testing.assert_approx_equal(distribution[1], 0.75)


@pytest.mark.parametrize('clusters_density,sparsity_based', [
    ({0: 5.0, 1: 15.0, 2: 25.0}, True),
    ({0: 5.0, 1: 15.0, 2: 25.0}, False)
])
def test_inter_distribute(clusters_density, sparsity_based):
    """Test the distribution of generated samples in each cluster."""
    clusterer = mock.Mock()
    clusterer.neighbors_ = [(0, 1), (0, 2), (0, 3)]
    distribution = dict(_inter_distribute(clusterer, clusters_density, sparsity_based, 0.0))
    if sparsity_based:
        np.testing.assert_approx_equal(distribution[(0, 1)], 0.6)
        np.testing.assert_approx_equal(distribution[(0, 2)], 0.4)
    else:
        np.testing.assert_approx_equal(distribution[(0, 1)], 0.4)
        np.testing.assert_approx_equal(distribution[(0, 2)], 0.6)


def test_filtering_threshold_parameter():
    """Test the filtering threshold parameter."""
    clusterer = mock.Mock()
    with pytest.raises(ValueError):
        distribution = density_distribution(clusterer, X, y, filtering_threshold=-1.0)
    with pytest.raises(TypeError):
        distribution = density_distribution(clusterer, X, y, filtering_threshold=None)


def test_distances_exponent_parameter():
    """Test the distances exponent parameter."""
    clusterer = mock.Mock()
    with pytest.raises(ValueError):
        distribution = density_distribution(clusterer, X, y, distances_exponent=-1.0)
    with pytest.raises(TypeError):
        distribution = density_distribution(clusterer, X, y, distances_exponent='value')


def test_sparsity_based_parameter():
    """Test the sparsity based parameter."""
    clusterer = mock.Mock()
    with pytest.raises(TypeError):
        distribution = density_distribution(clusterer, X, y, sparsity_based='value')


@pytest.mark.parametrize('distribution_ratio', [-1.0, 2.0])
def test_distribution_ratio_parameter(distribution_ratio):
    """Test the distribution ratio parameter."""
    clusterer = mock.Mock()
    clusterer.neighbors_ = NEIGHBORS
    with pytest.raises(ValueError):
        distribution = density_distribution(clusterer, X, y, distribution_ratio=distribution_ratio)


def test_distribution_ratio_parameter_neighbor():
    """Test the distribution ratio parameter."""
    clusterer = KMeans().fit(X, y)
    with pytest.raises(ValueError):
        distribution = density_distribution(clusterer, X, y, distribution_ratio=0.5)


@pytest.mark.parametrize('categorical_cols', [[], (), {}, 'value'])
def test_validate_categorical_cols_type(categorical_cols):
    """Test the type validation of categorical columns for the extended base oversampler."""
    oversampler = _TrivialOversampler(categorical_cols=categorical_cols)
    with pytest.raises(TypeError):    
        oversampler.fit(X, y)


@pytest.mark.parametrize('categorical_cols', [[-1], [0, X.shape[1]]])
def test_validate_categorical_cols_value(categorical_cols):
    """Test the value validation of categorical columns for the extended base oversampler."""
    oversampler = _TrivialOversampler(categorical_cols=categorical_cols)
    with pytest.raises(ValueError):    
        oversampler.fit(X, y)


@pytest.mark.parametrize('clusterer', [None, KMeans(), SOM()])
def test_fit(clusterer):
    """Test the fit method of the extended base oversampler."""
    oversampler = _TrivialOversampler(clusterer=clusterer).fit(X, y)
    assert oversampler.ratio_ == {0: 0, 1: 4, 2: 6}
    assert hasattr(oversampler, 'intra_distribution_')
    assert hasattr(oversampler, 'inter_distribution_')
    if isinstance(clusterer, SOM):
        assert len(oversampler.inter_distribution_) > 0
    else:
        assert len(oversampler.inter_distribution_) == 0


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
    clusterer = mock.Mock(spec=['fit', 'labels_'])
    clusterer.labels_ = LABELS
    oversampler = SMOTE(clusterer=clusterer)
    oversampler.fit(X, y, filtering_threshold=3.0, 
                    distances_exponent=0, sparsity_based=False)
    X_new, y_new = oversampler._intra_sample(X, y, oversampler.ratio_)
    assert X_new.shape == (9, X.shape[1])


def test_inter_sample():
    """Test the _inter_sample method of the extended base oversampler."""
    clusterer = mock.Mock(spec=['fit', 'labels_', 'neighbors_'])
    clusterer.labels_, clusterer.neighbors_  = LABELS, NEIGHBORS
    oversampler = SMOTE(clusterer=clusterer)
    oversampler.fit(X, y, filtering_threshold=3.0, distances_exponent=0, 
                    sparsity_based=False, distribution_ratio=0.0)
    initial_ratio = oversampler.ratio_
    k_neighbors = oversampler.k_neighbors
    X_new, y_new = oversampler._inter_sample(X, y, initial_ratio)
    assert oversampler.ratio_ == initial_ratio
    assert oversampler.k_neighbors == k_neighbors
    assert X_new.shape == (7, X.shape[1])
    

    
