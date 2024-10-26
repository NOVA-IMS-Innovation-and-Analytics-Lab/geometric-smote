"""Test the _base module."""

import numpy as np
import pytest
from imblearn_extra.clover.distribution.base import BaseDistributor
from sklearn.datasets import make_classification


@pytest.mark.parametrize(("n_samples", "n_classes", "weights"), [(20, 2, [0.8, 0.2]), (10, 3, [0.6, 0.2, 0.2])])
def test_fit(n_samples, n_classes, weights):
    """Test fit method."""
    X, y = make_classification(
        random_state=0,
        n_samples=n_samples,
        n_classes=n_classes,
        weights=weights,
        n_informative=5,
    )
    distributor = BaseDistributor().fit(X, y)
    assert len(distributor.majority_class_labels_) == 1
    assert distributor.majority_class_labels_[0] == 0
    np.testing.assert_array_equal(distributor.labels_, np.repeat(0, n_samples))
    np.testing.assert_array_equal(distributor.neighbors_, np.empty((0, 2)))
    assert distributor.intra_distribution_ == {(0, class_label): 1.0 for class_label in range(1, n_classes)}
    assert distributor.inter_distribution_ == {}
