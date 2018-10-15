"""
Test the distributor module.
"""

from itertools import product

import pytest
import numpy as np

from ..distribution import DensityDistributor

X = np.array(list(product(range(5), range(4))))
y = np.array([0] * 10 + [1] * 6 + [2] * 4)


@pytest.mark.parametrize('filtering_threshold,distances_exponent', [
    (1.0, 0),
    (1.0, 1),
    (2.0, 0),
    (2.0, 1)
])
def test_calculate_clusters_density(filtering_threshold, distances_exponent):
    """Test the calculation of filtered clusters density."""
    labels = np.array([0, 1, 1, 2, 2, 2, 2, 2])
    X = np.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0], [2, 0], [3, 0], [4, 0]])
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    distributor = DensityDistributor(labels=labels, filtering_threshold=filtering_threshold, distances_exponent=distances_exponent)
    distributor.fit(X, y)
    if filtering_threshold == 1.0:
        assert distributor.clusters_density_ == {}
    elif filtering_threshold == 2.0 and distances_exponent == 0:
        assert distributor.clusters_density_ == {1: 2.0, 2: 2.0}
    elif filtering_threshold == 1.0 and distances_exponent == 1:
        assert distributor.clusters_density_ == {1: 1.0, 2: 1.0}


@pytest.mark.parametrize('clusters_density,sparsity_based', [
    ({0: 5.0, 1: 15.0}, True),
    ({0: 5.0, 1: 15.0}, False)
])
def test_intra_distribute(clusters_density, sparsity_based):
    """Test the distribution of generated samples in each cluster."""
    distributor = DensityDistributor(sparsity_based=sparsity_based)
    distributor.clusters_density_, distributor.distribution_ratio_ = clusters_density, 1.0
    distributor._intra_distribute()
    distribution = dict(distributor.intra_distribution_)
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
    distributor = DensityDistributor(neighbors=[(0, 1), (0, 2), (0, 3)], sparsity_based=sparsity_based)
    distributor.clusters_density_, distributor.distribution_ratio_ = clusters_density, 0.0
    distributor._inter_distribute()
    distribution = dict(distributor.inter_distribution_)

    if sparsity_based:
        np.testing.assert_approx_equal(distribution[(0, 1)], 0.6)
        np.testing.assert_approx_equal(distribution[(0, 2)], 0.4)
    else:
        np.testing.assert_approx_equal(distribution[(0, 1)], 0.4)
        np.testing.assert_approx_equal(distribution[(0, 2)], 0.6)


def test_filtering_threshold_parameter():
    """Test the filtering threshold parameter."""
    with pytest.raises(ValueError):
        distributor = DensityDistributor(filtering_threshold=-1.0).fit(X, y)
    with pytest.raises(TypeError):
        distributor = DensityDistributor(filtering_threshold=None).fit(X, y)


def test_distances_exponent_parameter():
    """Test the distances exponent parameter."""
    with pytest.raises(ValueError):
        distributor = DensityDistributor(distances_exponent=-1.0).fit(X, y)
    with pytest.raises(TypeError):
        distributor = DensityDistributor(distances_exponent=None).fit(X, y)


def test_sparsity_based_parameter():
    """Test the sparsity based parameter."""
    with pytest.raises(TypeError):
        distributor = DensityDistributor(sparsity_based=None).fit(X, y)


def test_distribution_ratio_parameter():
    """Test the distribution ratio parameter."""
    with pytest.raises(ValueError):
        distributor = DensityDistributor(distribution_ratio=-1.0).fit(X, y)
    with pytest.raises(ValueError):
        distributor = DensityDistributor(distribution_ratio=2.0).fit(X, y)
    with pytest.raises(TypeError):
        distributor = DensityDistributor(distribution_ratio='value').fit(X, y)


def test_distribution_ratio_parameter_neighbor():
    """Test the distribution ratio parameter."""
    with pytest.raises(ValueError):
        distributor = DensityDistributor(neighbors=None, distribution_ratio=0.5).fit(X, y)


def test_fit():
    """Test the fit method."""
    labels = np.array([0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2, 0, 3, 3, 3, 0, 3, 3, 3])
    neighbors = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    distributor = DensityDistributor(labels=labels, neighbors=neighbors, filtering_threshold=3.0, distribution_ratio=0.5, sparsity_based=False).fit(X, y)
    intra_distribution, inter_distribution = dict(distributor.intra_distribution_), dict(distributor.inter_distribution_)
    np.testing.assert_approx_equal(intra_distribution[0], 0.1)
    np.testing.assert_approx_equal(intra_distribution[2], 0.1)
    np.testing.assert_approx_equal(intra_distribution[3], 0.3)
    np.testing.assert_approx_equal(inter_distribution[(0, 2)], 0.1)
    np.testing.assert_approx_equal(inter_distribution[(0, 3)], 0.2)
    np.testing.assert_approx_equal(inter_distribution[(2, 3)], 0.2) 