"""Test the _density module."""

import numpy as np
import pytest
from imblearn_extra.clover.distribution._density import DensityDistributor
from sklearn.base import clone

X = np.array(
    [
        [1.0, 1.0],
        [1.0, 2.0],
        [1.5, 1.5],
        [-1.0, 1.0],
        [-1.0, 1.5],
        [-1.0, -1.0],
        [2.0, -1.0],
        [2.5, -1.0],
        [2.5, -1.5],
        [2.0, -1.5],
        [2.0, -2.0],
        [2.0, -2.5],
        [3.0, -1.0],
        [2.0, -1.0],
        [4.0, -1.0],
    ],
)
y_bin = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])
y_multi = np.array([0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2])
y_partial_tie = np.array([0, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 2])
y_full_tie = np.array([0, 1, 2, 1, 2, 1, 2, 2, 0, 0, 0, 0, 1, 1, 2])
LABELS = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4])
NEIGHBORS_BIN = np.array([(0, 1), (0, 2), (0, 3), (4, 2), (2, 3)])
NEIGHBORS_MULTI = np.array([(0, 1), (1, 4), (2, 3)])
DISTRIBUTOR = DensityDistributor(filtering_threshold=0.6, distances_exponent=1)


def test_filtered_clusters_binary():
    """Test the identification of filtered clusters.

    Binary case.
    """
    distributor = clone(DISTRIBUTOR).fit(X, y_bin, LABELS)
    assert distributor.filtered_clusters_ == [(0, 1), (2, 1), (4, 1)]


def test_filtered_clusters_multiclass():
    """Test the identification of filtered clusters.

    Multiclass case.
    """
    distributor = clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_multi, LABELS)
    assert distributor.filtered_clusters_ == [
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (4, 1),
        (4, 2),
    ]


def test_filtered_clusters_multiclass_partial_tie():
    """Test the identification of filtered clusters.

    Multiclass case with partial tie.
    """
    distributor = clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_partial_tie, LABELS)
    assert distributor.filtered_clusters_ == [(1, 2), (4, 2)]


def test_filtered_clusters_multiclass_full_tie():
    """Test the identification of filtered clusters.

    Multiclass case with full tie.
    """
    distributor = clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_full_tie, LABELS)
    assert distributor.filtered_clusters_ == []


def test_clusters_density_binary():
    """Test the filtered clusters density.

    Binary case.
    """
    distributor = clone(DISTRIBUTOR).fit(X, y_bin, LABELS)
    assert distributor.clusters_density_ == {(0, 1): 2.0, (2, 1): 2.25, (4, 1): 2.25}


def test_clusters_density_multiclass():
    """Test the filtered clusters density.

    Multiclass case.
    """
    distributor = clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_multi, LABELS)
    assert distributor.clusters_density_ == {
        (0, 1): 2.0,
        (0, 2): 2.0,
        (1, 1): 2.0,
        (1, 2): 2.0,
        (4, 1): 2.0,
        (4, 2): 2.0,
    }


def test_clusters_density_multiclass_partial_tie():
    """Test filtered clusters density.

    Multiclass case with partial tie.
    """
    distributor = clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_partial_tie, LABELS)
    assert distributor.clusters_density_ == {
        (1, 2): 4.0,
        (4, 2): 4.0,
    }


def test_clusters_density_multiclass_full_tie():
    """Test filtered clusters density.

    Multiclass case with full tie.
    """
    distributor = clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_full_tie, LABELS)
    assert distributor.clusters_density_ == {}


def test_clusters_density_no_filtered():
    """Test filter clusters density.

    No filtered clusters case.
    """
    X = np.arange(0.0, 5.0).reshape(-1, 1)
    y = np.array([0, 0, 0, 1, 1])
    labels = np.array([-1, -1, -1, -1, -1])
    distributor = clone(DISTRIBUTOR).set_params().fit(X, y, labels)
    assert distributor.clusters_density_ == {}


def test_raise_error_filtering_threshold():
    """Test raise error for filtering threshold.

    Value and type error cases.
    """
    with pytest.raises(ValueError, match='filtering_threshold == -1.0, must be >= 0.0'):
        clone(DISTRIBUTOR).set_params(filtering_threshold=-1.0).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError, match='filtering_threshold must be an instance of {int, float}, not NoneType'):
        clone(DISTRIBUTOR).set_params(filtering_threshold=None).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError, match='filtering_threshold must be an instance of {int, float}, not str'):
        clone(DISTRIBUTOR).set_params(filtering_threshold='value').fit(X, y_bin, LABELS)


def test_raise_error_distances_exponent():
    """Test raise error for distances exponent.

    Value and type error cases.
    """
    with pytest.raises(ValueError, match='distances_exponent == -1.0, must be >= 0.0'):
        clone(DISTRIBUTOR).set_params(distances_exponent=-1.0).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError, match='distances_exponent must be an instance of {int, float}, not None'):
        clone(DISTRIBUTOR).set_params(distances_exponent=None).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError, match='distances_exponent must be an instance of {int, float}, not str'):
        clone(DISTRIBUTOR).set_params(distances_exponent='value').fit(X, y_bin, LABELS)


def test_raise_error_sparsity_based():
    """Test raise error for sparsity based.

    Type error case.
    """
    with pytest.raises(TypeError, match='sparsity_based must be an instance of bool, not NoneType'):
        clone(DISTRIBUTOR).set_params(sparsity_based=None).fit(X, y_bin, LABELS)


def test_raise_error_distribution_ratio():
    """Test raise error for distribution ratio.

    Type error case.
    """
    with pytest.raises(ValueError, match='distribution_ratio == -1.0, must be >= 0.0'):
        clone(DISTRIBUTOR).set_params(distribution_ratio=-1.0).fit(X, y_bin, LABELS)
    with pytest.raises(ValueError, match='distribution_ratio == 2.0, must be <= 1.0'):
        clone(DISTRIBUTOR).set_params(distribution_ratio=2.0).fit(X, y_bin, LABELS)
    with pytest.raises(TypeError, match='distribution_ratio must be an instance of float, not str'):
        clone(DISTRIBUTOR).set_params(distribution_ratio='value').fit(X, y_bin, LABELS)


def test_raise_error_no_neighbors_distribution_ratio():
    """Test distribution ratio.

    No neighbors value error case.
    """
    with pytest.raises(
        ValueError,
        match=('Parameter `distribution_ratio` should be equal to 1.0, when `neighbors` parameter is `None`.'),
    ):
        clone(DISTRIBUTOR).set_params(distribution_ratio=0.5).fit(X, y_bin, LABELS, neighbors=None)


def test_fit_default():
    """Test fit method.

    Default initialization case.
    """
    distributor = clone(DISTRIBUTOR).fit(X, y_bin, None, None)
    assert distributor.majority_class_labels_ == [0]
    assert hasattr(distributor, 'filtered_clusters_')
    assert hasattr(distributor, 'clusters_density_')
    np.testing.assert_array_equal(distributor.labels_, np.repeat(0, len(X)))
    np.testing.assert_array_equal(distributor.neighbors_, np.empty((0, 2)))
    assert distributor.intra_distribution_ == {(0, 1): 1.0}
    assert distributor.inter_distribution_ == {}


def test_fit_binary_intra():
    """Test fit method.

    Binary and intra-cluster generation case.
    """
    distributor = clone(DISTRIBUTOR).fit(X, y_bin, LABELS)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 1)], 9.0 / 25.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(2, 1)], 8.0 / 25.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 1)], 8.0 / 25.0)


def test_fit_multiclass_intra():
    """Test fit method.

    Multiclass and intra-cluster generation case.
    """
    distributor = clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_multi, LABELS)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 1)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 1)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 1)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 2)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 2)], 1.0 / 3.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 2)], 1.0 / 3.0)


def test_fit_multiclass_intra_partial_tie():
    """Test fit method.

    Multiclass intra-cluster generation and partial tie case.
    """
    distributor = clone(DISTRIBUTOR).set_params(filtering_threshold=1.0).fit(X, y_partial_tie, LABELS)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 2)], 0.5)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 2)], 0.5)


def test_fit_binary_inter():
    """Test fit method.

    Binary and inter-cluster generation case.
    """
    distributor = clone(DISTRIBUTOR).set_params(distribution_ratio=0.0).fit(X, y_bin, LABELS, NEIGHBORS_BIN)
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_BIN)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((0, 1), (2, 1))], 18.0 / 35.0)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((4, 1), (2, 1))], 17.0 / 35.0)


def test_fit_multiclass_inter():
    """Test fit method.

    Multiclass and inter-cluster generation case.
    """
    distributor = (
        clone(DISTRIBUTOR)
        .set_params(distribution_ratio=0.0, filtering_threshold=1.0)
        .fit(X, y_multi, LABELS, NEIGHBORS_MULTI)
    )
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_MULTI)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((0, 1), (1, 1))], 0.5)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((1, 1), (4, 1))], 0.5)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((0, 2), (1, 2))], 0.5)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((1, 2), (4, 2))], 0.5)


def test_fit_multiclass_inter_partial_tie():
    """Test fit method.

    Multiclass, intra-cluster generation and partial tie case.
    """
    distributor = (
        clone(DISTRIBUTOR)
        .set_params(distribution_ratio=0.0, filtering_threshold=1.0)
        .fit(X, y_partial_tie, LABELS, NEIGHBORS_MULTI)
    )
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_MULTI)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((1, 2), (4, 2))], 1)


def test_fit_binary_intra_inter():
    """Test fit method.

    Binary, intra-cluster generation and inter-cluster generation case.
    """
    distributor = clone(DISTRIBUTOR).set_params(distribution_ratio=0.5).fit(X, y_bin, LABELS, NEIGHBORS_BIN)
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_BIN)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 1)], 9.0 / 50.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(2, 1)], 8.0 / 50.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 1)], 8.0 / 50.0)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((0, 1), (2, 1))], 18.0 / 70.0)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((4, 1), (2, 1))], 17.0 / 70.0)


def test_fit_multiclass_intra_inter():
    """Test fit method.

    Multiclass, intra-cluster generation and inter-cluster generation
    case.
    """
    distributor = (
        clone(DISTRIBUTOR)
        .set_params(distribution_ratio=0.5, filtering_threshold=1.0)
        .fit(X, y_multi, LABELS, NEIGHBORS_MULTI)
    )
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_MULTI)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 1)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 1)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 1)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(0, 2)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 2)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 2)], 1.0 / 6.0)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((0, 1), (1, 1))], 0.25)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((1, 1), (4, 1))], 0.25)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((0, 2), (1, 2))], 0.25)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((1, 2), (4, 2))], 0.25)


def test_fit_multiclass_intra_inter_partial_tie():
    """Test fit method.

    Multiclass, intra-cluster generation, inter-cluster generation case
    and partial tie case.
    """
    distributor = (
        clone(DISTRIBUTOR)
        .set_params(distribution_ratio=0.5, filtering_threshold=1.0)
        .fit(X, y_partial_tie, LABELS, NEIGHBORS_MULTI)
    )
    np.testing.assert_equal(distributor.labels_, LABELS)
    np.testing.assert_equal(distributor.neighbors_, NEIGHBORS_MULTI)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(1, 2)], 1.0 / 4.0)
    np.testing.assert_almost_equal(distributor.intra_distribution_[(4, 2)], 1.0 / 4.0)
    np.testing.assert_almost_equal(distributor.inter_distribution_[((1, 2), (4, 2))], 0.5)
