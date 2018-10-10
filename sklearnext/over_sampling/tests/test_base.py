"""
Test the base module.
"""

from collections import OrderedDict
from unittest import mock

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from imblearn.utils import check_ratio

from ...cluster import SOM
from ..base import (
    _count_clusters_samples,
    _calculate_clusters_density,
    _intra_distribute,
    _inter_distribute,
    density_distribution,
    ExtendedBaseOverSampler
)

X_bin, y_bin = make_classification(random_state=0, weights=[0.9, 0.1])
X_multi, y_multi = make_classification(random_state=0, n_classes=3, n_informative=5, weights=[0.7, 0.2, 0.1])


class _TestOverSampler(ExtendedBaseOverSampler):
    """Oversampler used for testing."""

    def _numerical_sample(self, X, y):
        pass


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
        distribution = density_distribution(clusterer, X_bin, y_bin, filtering_threshold=-1.0)
    with pytest.raises(TypeError):
        distribution = density_distribution(clusterer, X_bin, y_bin, filtering_threshold=None)


def test_distances_exponent_parameter():
    """Test the distances exponent parameter."""
    clusterer = mock.Mock()
    with pytest.raises(ValueError):
        distribution = density_distribution(clusterer, X_bin, y_bin, distances_exponent=-1.0)
    with pytest.raises(TypeError):
        distribution = density_distribution(clusterer, X_bin, y_bin, distances_exponent='value')


def test_sparsity_based_parameter():
    """Test the sparsity based parameter."""
    clusterer = mock.Mock()
    with pytest.raises(TypeError):
        distribution = density_distribution(clusterer, X_bin, y_bin, sparsity_based='value')


@pytest.mark.parametrize('distribution_ratio', [-1.0, 2.0])
def test_distribution_ratio_parameter(distribution_ratio):
    """Test the distribution ratio parameter."""
    clusterer = mock.Mock()
    clusterer.neighbors_ = [(0, 1), (0, 2), (0, 3)]
    with pytest.raises(ValueError):
        distribution = density_distribution(clusterer, X_bin, y_bin, distribution_ratio=distribution_ratio)


def test_distribution_ratio_parameter_neighbor():
    """Test the distribution ratio parameter."""
    clusterer = KMeans()
    with pytest.raises(ValueError):
        distribution = density_distribution(clusterer, X_bin, y_bin, distribution_ratio=0.5)


@pytest.mark.parametrize('categorical_cols', [[], (), {}, 'value'])
def test_validate_categorical_cols_type(categorical_cols):
    """Test the type validation of categorical columns for the extended base oversampler."""
    oversampler = _TestOverSampler(categorical_cols=categorical_cols)
    with pytest.raises(TypeError):    
        oversampler.fit(X_bin, y_bin)


@pytest.mark.parametrize('categorical_cols', [[-1], [0, X_bin.shape[1]]])
def test_validate_categorical_cols_value(categorical_cols):
    """Test the value validation of categorical columns for the extended base oversampler."""
    oversampler = _TestOverSampler(categorical_cols=categorical_cols)
    with pytest.raises(ValueError):    
        oversampler.fit(X_bin, y_bin)
