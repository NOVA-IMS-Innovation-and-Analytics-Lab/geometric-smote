"""Test the _gsomo module."""

from collections import Counter, OrderedDict
from math import sqrt

import pytest
from imblearn_extra.clover.clusterer import SOM
from imblearn_extra.clover.distribution import DensityDistributor
from imblearn_extra.clover.over_sampling import GeometricSOMO
from imblearn_extra.gsmote import GeometricSMOTE
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification

RANDOM_STATE = 11
X, y = make_classification(
    random_state=RANDOM_STATE,
    n_classes=3,
    n_samples=5000,
    n_features=10,
    n_clusters_per_class=2,
    weights=[0.3, 0.45, 0.25],
    n_informative=5,
)
GSOMO_OVERSAMPLER = GeometricSOMO(random_state=RANDOM_STATE)


@pytest.mark.parametrize(
    ('k_neighbors', 'imbalance_ratio_threshold', 'distances_exponent', 'distribution_ratio'),
    [(3, 2.0, 'auto', 0.3), (5, 1.5, 8, 0.5), (8, 'auto', 10, 0.8)],
)
def test_fit(k_neighbors, imbalance_ratio_threshold, distances_exponent, distribution_ratio):
    """Test fit method.

    Multiple cases.
    """
    # Fit oversampler
    params = {
        'k_neighbors': k_neighbors,
        'distribution_ratio': distribution_ratio,
        'distances_exponent': distances_exponent,
        'imbalance_ratio_threshold': imbalance_ratio_threshold,
    }
    gsomo = clone(GSOMO_OVERSAMPLER).set_params(**params).fit(X, y)
    y_count = Counter(y)

    # Assert random state
    assert hasattr(gsomo, 'random_state_')

    # Assert oversampler
    assert isinstance(gsomo.oversampler_, GeometricSMOTE)
    assert gsomo.oversampler_.k_neighbors == gsomo.k_neighbors == k_neighbors
    assert gsomo.oversampler_.truncation_factor == gsomo.truncation_factor
    assert gsomo.oversampler_.deformation_factor == gsomo.deformation_factor
    assert gsomo.oversampler_.selection_strategy == gsomo.selection_strategy

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, SOM)

    # Assert distributor
    assert isinstance(gsomo.distributor_, DensityDistributor)
    assert gsomo.distributor_.filtering_threshold == gsomo.imbalance_ratio_threshold == imbalance_ratio_threshold
    assert gsomo.distributor_.distances_exponent == gsomo.distances_exponent == distances_exponent
    assert gsomo.distributor_.distribution_ratio == gsomo.distribution_ratio == distribution_ratio

    # Assert sampling strategy
    assert gsomo.oversampler_.sampling_strategy == gsomo.sampling_strategy
    assert gsomo.sampling_strategy_ == OrderedDict({0: y_count[1] - y_count[0], 2: y_count[1] - y_count[2]})


def test_fit_default():
    """Test fit method.

    Default case.
    """
    # Fit oversampler
    gsomo = clone(GSOMO_OVERSAMPLER).fit(X, y)

    # Create SOM instance with default parameters
    som = SOM()

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, SOM)
    assert gsomo.clusterer_.n_rows == som.n_rows
    assert gsomo.clusterer_.n_columns == som.n_columns


@pytest.mark.parametrize('n_clusters', [5, 6, 12])
def test_fit_number_of_clusters(n_clusters):
    """Test clusterer of fit method.

    Number of clusters case.
    """
    # Fit oversampler
    gsomo = clone(GSOMO_OVERSAMPLER).set_params(som_estimator=n_clusters).fit(X, y)

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, SOM)
    assert gsomo.clusterer_.n_rows == round(sqrt(gsomo.som_estimator))
    assert gsomo.clusterer_.n_columns == round(sqrt(gsomo.som_estimator))


@pytest.mark.parametrize('proportion', [0.0, 0.5, 1.0])
def test_fit_proportion_of_samples(proportion):
    """Test clusterer of fit method.

    Proportion of samples case.
    """
    # Fit oversampler
    gsomo = clone(GSOMO_OVERSAMPLER).set_params(som_estimator=proportion).fit(X, y)

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, SOM)
    assert gsomo.clusterer_.n_rows == round(sqrt((X.shape[0] - 1) * gsomo.som_estimator + 1))
    assert gsomo.clusterer_.n_columns == round(sqrt((X.shape[0] - 1) * gsomo.som_estimator + 1))


def test_som_estimator():
    """Test clusterer of fit method.

    Clusterer case.
    """
    # Fit oversampler
    gsomo = clone(GSOMO_OVERSAMPLER).set_params(som_estimator=SOM()).fit(X, y)

    # Define som estimator
    som = SOM()

    # Assert clusterer
    assert isinstance(gsomo.clusterer_, type(som))
    assert gsomo.clusterer_.n_rows == som.n_rows
    assert gsomo.clusterer_.n_columns == som.n_columns


@pytest.mark.parametrize('som_estimator', [-3, 0])
def test_raise_value_error_fit_integer(som_estimator):
    """Test fit method.

    Integer values as estimators error case.
    """
    with pytest.raises(ValueError, match=f'som_estimator == {som_estimator}, must be >= 1.'):
        clone(GSOMO_OVERSAMPLER).set_params(som_estimator=som_estimator).fit(X, y)


@pytest.mark.parametrize('som_estimator', [-1.5, 2.0])
def test_raise_value_error_fit_float(som_estimator):
    """Test fit method.

    Float values as estimators error case.
    """
    with pytest.raises(ValueError, match=f'som_estimator == {som_estimator}, must be'):
        clone(GSOMO_OVERSAMPLER).set_params(som_estimator=som_estimator).fit(X, y)


@pytest.mark.parametrize('som_estimator', [AgglomerativeClustering, [3, 5]])
def test_raise_type_error_fit(som_estimator):
    """Test fit method.

    Not SOMO clusterer error case.
    """
    with pytest.raises(TypeError, match='Parameter `som_estimator` should be'):
        clone(GSOMO_OVERSAMPLER).set_params(som_estimator=som_estimator).fit(X, y)


def test_fit_resample():
    """Test fit and resample method.

    Default case.
    """
    # Fit oversampler
    gsomo = clone(GSOMO_OVERSAMPLER)
    _, y_res = gsomo.fit_resample(X, y)

    # Assert clusterer is fitted
    assert hasattr(gsomo.clusterer_, 'labels_')
    assert hasattr(gsomo.clusterer_, 'neighbors_')

    # Assert distributor is fitted
    assert hasattr(gsomo.distributor_, 'intra_distribution_')
    assert hasattr(gsomo.distributor_, 'inter_distribution_')
