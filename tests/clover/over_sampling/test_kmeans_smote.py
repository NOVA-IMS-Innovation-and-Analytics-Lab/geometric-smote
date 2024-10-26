"""Test the _kmeans_smote module."""

from collections import Counter, OrderedDict

import pytest
from imblearn.over_sampling import SMOTE
from imblearn_extra.clover.distribution import DensityDistributor
from imblearn_extra.clover.over_sampling import KMeansSMOTE
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.datasets import make_classification

RANDOM_STATE = 1
X, y = make_classification(
    random_state=RANDOM_STATE,
    n_classes=3,
    n_samples=5000,
    n_features=10,
    n_clusters_per_class=2,
    weights=[0.25, 0.45, 0.3],
    n_informative=5,
)
KMEANS_SMOTE_OVERSAMPLER = KMeansSMOTE(random_state=RANDOM_STATE)


@pytest.mark.parametrize(
    ('k_neighbors', 'imbalance_ratio_threshold', 'distances_exponent'),
    [(3, 2.0, 'auto'), (5, 1.5, 8), (8, 'auto', 10)],
)
def test_fit(k_neighbors, imbalance_ratio_threshold, distances_exponent):
    """Test fit method.

    Multiple cases.
    """
    # Fit oversampler
    params = {
        'k_neighbors': k_neighbors,
        'imbalance_ratio_threshold': imbalance_ratio_threshold,
        'distances_exponent': distances_exponent,
    }
    kmeans_smote = clone(KMEANS_SMOTE_OVERSAMPLER).set_params(**params).fit(X, y)
    y_count = Counter(y)

    # Assert random state
    assert hasattr(kmeans_smote, 'random_state_')

    # Assert oversampler
    assert isinstance(kmeans_smote.oversampler_, SMOTE)
    assert kmeans_smote.oversampler_.k_neighbors == kmeans_smote.k_neighbors == k_neighbors

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, MiniBatchKMeans)

    # Assert distributor
    assert isinstance(kmeans_smote.distributor_, DensityDistributor)
    assert (
        kmeans_smote.distributor_.filtering_threshold
        == kmeans_smote.imbalance_ratio_threshold
        == imbalance_ratio_threshold
    )
    assert kmeans_smote.distributor_.distances_exponent == kmeans_smote.distances_exponent == distances_exponent

    # Assert sampling strategy
    assert kmeans_smote.oversampler_.sampling_strategy == kmeans_smote.sampling_strategy
    assert kmeans_smote.sampling_strategy_ == OrderedDict({0: y_count[1] - y_count[0], 2: y_count[1] - y_count[2]})


def test_fit_default():
    """Test fit method.

    Default case.
    """
    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE_OVERSAMPLER).fit(X, y)

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, MiniBatchKMeans)
    assert kmeans_smote.clusterer_.n_clusters == MiniBatchKMeans().n_clusters


@pytest.mark.parametrize('n_clusters', [5, 6, 12])
def test_fit_number_of_clusters(n_clusters):
    """Test fit method.

    Number of clusters case.
    """
    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE_OVERSAMPLER).set_params(kmeans_estimator=n_clusters).fit(X, y)

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, MiniBatchKMeans)
    assert kmeans_smote.clusterer_.n_clusters == n_clusters


@pytest.mark.parametrize('proportion', [0.0, 0.5, 1.0])
def test_fit_proportion_of_samples(proportion):
    """Test fit method.

    Proportion of samples case.
    """
    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE_OVERSAMPLER).set_params(kmeans_estimator=proportion).fit(X, y)

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, MiniBatchKMeans)
    assert kmeans_smote.clusterer_.n_clusters == round((len(X) - 1) * proportion + 1)


@pytest.mark.parametrize('kmeans_estimator', [KMeans(), MiniBatchKMeans()])
def test_fit_kmeans_estimator(kmeans_estimator):
    """Test fit method.

    KMeans estimator case.
    """
    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE_OVERSAMPLER).set_params(kmeans_estimator=kmeans_estimator).fit(X, y)

    # Assert clusterer
    assert isinstance(kmeans_smote.clusterer_, type(kmeans_estimator))
    assert kmeans_smote.clusterer_.n_clusters == kmeans_estimator.n_clusters


@pytest.mark.parametrize('kmeans_estimator', [-3, 0])
def test_raise_value_error_fit_integer(kmeans_estimator):
    """Test fit method.

    Integer values as estimators error case.
    """
    with pytest.raises(ValueError, match=f'kmeans_estimator == {kmeans_estimator}, must be >= 1.'):
        clone(KMEANS_SMOTE_OVERSAMPLER).set_params(kmeans_estimator=kmeans_estimator).fit(X, y)


@pytest.mark.parametrize('kmeans_estimator', [-1.5, 2.0])
def test_raise_value_error_fit_float(kmeans_estimator):
    """Test fit method.

    Float values as estimators error case.
    """
    with pytest.raises(ValueError, match=f'kmeans_estimator == {kmeans_estimator}, must be'):
        clone(KMEANS_SMOTE_OVERSAMPLER).set_params(kmeans_estimator=kmeans_estimator).fit(X, y)


@pytest.mark.parametrize('kmeans_estimator', [AgglomerativeClustering(), [3, 5]])
def test_raise_type_error_fit(kmeans_estimator):
    """Test fit method.

    Not KMeans clusterer error case.
    """
    with pytest.raises(TypeError, match='Parameter `kmeans_estimator` should be'):
        clone(KMEANS_SMOTE_OVERSAMPLER).set_params(kmeans_estimator=kmeans_estimator).fit(X, y)


def test_fit_resample():
    """Test fit and resample method.

    Default case.
    """
    # Fit oversampler
    kmeans_smote = clone(KMEANS_SMOTE_OVERSAMPLER)
    _, y_res = kmeans_smote.fit_resample(X, y)

    # Assert clusterer is fitted
    assert hasattr(kmeans_smote.clusterer_, 'labels_')
    assert not hasattr(kmeans_smote.clusterer_, 'neighbors_')

    # Assert distributor is fitted
    assert hasattr(kmeans_smote.distributor_, 'intra_distribution_')
    assert hasattr(kmeans_smote.distributor_, 'inter_distribution_')
