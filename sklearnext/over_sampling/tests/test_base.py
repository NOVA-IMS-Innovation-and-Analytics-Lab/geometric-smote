"""
Test the base module.
"""

from collections import OrderedDict

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from imblearn.utils import check_ratio

from ..base import (
    _count_clusters_samples,
    _calculate_clusters_density,
    _intra_distribute,
    _inter_distribute,
    generate_distribution,
    ExtendedBaseOverSampler
)

X_bin, y_bin = make_classification(random_state=0, weights=[0.9, 0.1])
X_multi, y_multi = make_classification(random_state=0, n_classes=3, n_informative=5, weights=[0.7, 0.2, 0.1])


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
    distribution = _intra_distribute(clusters_density, sparsity_based, 1.0)
    if clusters_density == {0: 5.0, 1: 15.0} and sparsity_based:
        assert distribution == [(0, 0.75), (1, 0.25)]
    elif clusters_density == {0: 5.0, 1: 15.0} and not sparsity_based:
        assert distribution == [(0, 0.25), (1, 0.75)]


@pytest.mark.parametrize('clusters_density,sparsity_based', [
    ({0: 5.0, 1: 15.0}, True),
    ({0: 5.0, 1: 15.0}, False)
])
def test_inter_distribute(clusters_density, sparsity_based):
    """Test the distribution of generated samples in each cluster."""
    pass
    



