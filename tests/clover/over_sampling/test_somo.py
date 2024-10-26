"""Test the _somo module."""

from collections import Counter, OrderedDict
from math import sqrt

import pytest
from imblearn.over_sampling import SMOTE
from imblearn_extra.clover.clusterer import SOM
from imblearn_extra.clover.distribution import DensityDistributor
from imblearn_extra.clover.over_sampling import SOMO
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification

RANDOM_STATE = 2
X, y = make_classification(
    random_state=RANDOM_STATE,
    n_classes=3,
    n_samples=5000,
    n_features=10,
    n_clusters_per_class=2,
    weights=[0.25, 0.45, 0.3],
    n_informative=5,
)
SOMO_OVERSAMPLER = SOMO(random_state=RANDOM_STATE)


@pytest.mark.parametrize(
    ('k_neighbors', 'distribution_ratio'),
    [(3, 0.2), (5, 0.5), (8, 0.6)],
)
def test_fit(k_neighbors, distribution_ratio):
    """Test fit method.

    Multiple cases.
    """
    # Fit oversampler
    params = {'k_neighbors': k_neighbors, 'distribution_ratio': distribution_ratio}
    somo = clone(SOMO_OVERSAMPLER).set_params(**params).fit(X, y)
    y_count = Counter(y)

    # Assert random state
    assert hasattr(somo, 'random_state_')

    # Assert oversampler
    assert isinstance(somo.oversampler_, SMOTE)
    assert somo.oversampler_.k_neighbors == somo.k_neighbors == k_neighbors

    # Assert clusterer
    assert isinstance(somo.clusterer_, SOM)

    # Assert distributor
    filtering_threshold = 1.0
    distances_exponent = 2
    assert isinstance(somo.distributor_, DensityDistributor)
    assert somo.distributor_.filtering_threshold == filtering_threshold
    assert somo.distributor_.distances_exponent == distances_exponent
    assert somo.distributor_.distribution_ratio == somo.distribution_ratio == distribution_ratio

    # Assert sampling strategy
    assert somo.oversampler_.sampling_strategy == somo.sampling_strategy
    assert somo.sampling_strategy_ == OrderedDict({0: y_count[1] - y_count[0], 2: y_count[1] - y_count[2]})


def test_fit_default():
    """Test fit method.

    Default case.
    """
    # Fit oversampler
    somo = clone(SOMO_OVERSAMPLER).fit(X, y)

    # Create SOM instance with default parameters
    som = SOM()

    # Assert clusterer
    assert isinstance(somo.clusterer_, SOM)
    assert somo.clusterer_.n_rows == som.n_rows
    assert somo.clusterer_.n_columns == som.n_columns


@pytest.mark.parametrize('n_clusters', [5, 6, 12])
def test_fit_number_of_clusters(n_clusters):
    """Test fit method.

    Number of clusters case.
    """
    # Fit oversampler
    somo = clone(SOMO_OVERSAMPLER).set_params(som_estimator=n_clusters).fit(X, y)

    # Assert clusterer
    assert isinstance(somo.clusterer_, SOM)
    assert somo.clusterer_.n_rows == round(sqrt(somo.som_estimator))
    assert somo.clusterer_.n_columns == round(sqrt(somo.som_estimator))


@pytest.mark.parametrize('proportion', [0.0, 0.5, 1.0])
def test_fit_proportion_of_samples(proportion):
    """Test fit method.

    Proportion of samples case.
    """
    # Fit oversampler
    somo = clone(SOMO_OVERSAMPLER).set_params(som_estimator=proportion).fit(X, y)

    # Assert clusterer
    assert isinstance(somo.clusterer_, SOM)
    assert somo.clusterer_.n_rows == round(sqrt((X.shape[0] - 1) * somo.som_estimator + 1))
    assert somo.clusterer_.n_columns == round(sqrt((X.shape[0] - 1) * somo.som_estimator + 1))


def test_fit_som_estimator():
    """Test fit method.

    SOM estimator case.
    """
    # Fit oversampler
    somo = clone(SOMO_OVERSAMPLER).set_params(som_estimator=SOM()).fit(X, y)

    # Define som estimator
    som = SOM()

    # Assert clusterer
    assert isinstance(somo.clusterer_, type(som))
    assert somo.clusterer_.n_rows == som.n_rows
    assert somo.clusterer_.n_columns == som.n_columns


@pytest.mark.parametrize('som_estimator', [-3, 0])
def test_raise_value_error_fit_integer(som_estimator):
    """Test fit method.

    Integer values as estimators error case.
    """
    with pytest.raises(ValueError, match=f'som_estimator == {som_estimator}, must be >= 1.'):
        clone(SOMO_OVERSAMPLER).set_params(som_estimator=som_estimator).fit(X, y)


@pytest.mark.parametrize('som_estimator', [-1.5, 2.0])
def test_raise_value_error_fit_float(som_estimator):
    """Test fit method.

    Float values as estimators error case.
    """
    with pytest.raises(ValueError, match=f'som_estimator == {som_estimator}, must be'):
        clone(SOMO_OVERSAMPLER).set_params(som_estimator=som_estimator).fit(X, y)


@pytest.mark.parametrize('som_estimator', [AgglomerativeClustering(), [3, 5]])
def test_raise_type_error_fit(som_estimator):
    """Test fit method.

    Not SOM clusterer error case.
    """
    with pytest.raises(TypeError, match='Parameter `som_estimator` should be'):
        clone(SOMO_OVERSAMPLER).set_params(som_estimator=som_estimator).fit(X, y)


def test_fit_resample():
    """Test fit and resample method.

    Default case.
    """
    # Fit oversampler
    somo = clone(SOMO_OVERSAMPLER)
    _, y_res = somo.fit_resample(X, y)

    # Assert clusterer is fitted
    assert hasattr(somo.clusterer_, 'labels_')
    assert hasattr(somo.clusterer_, 'neighbors_')

    # Assert distributor is fitted
    assert hasattr(somo.distributor_, 'intra_distribution_')
    assert hasattr(somo.distributor_, 'inter_distribution_')
