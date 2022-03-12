"""
Test the geometric_smote module.
"""

from collections import Counter

import pytest
import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_classification
from scipy import sparse

from ..geometric_smote import _make_geometric_sample, GeometricSMOTE, SELECTION_STRATEGY

RND_SEED = 0
RANDOM_STATE = check_random_state(RND_SEED)
CENTERS = [
    RANDOM_STATE.random_sample((2,)),
    2.6 * RANDOM_STATE.random_sample((4,)),
    3.2 * RANDOM_STATE.random_sample((10,)),
    -0.5 * RANDOM_STATE.random_sample((1,)),
]
SURFACE_POINTS = [
    RANDOM_STATE.random_sample((2,)),
    5.2 * RANDOM_STATE.random_sample((4,)),
    -3.5 * RANDOM_STATE.random_sample((10,)),
    -10.9 * RANDOM_STATE.random_sample((1,)),
]
TRUNCATION_FACTORS = [-1.0, -0.5, 0.0, 0.5, 1.0]
DEFORMATION_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0]


def data_heterogeneous_ordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, :2] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 2] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    # return the categories
    return X, y, [2, 3]


def data_heterogeneous_unordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    # return the categories
    return X, y, [0, 3]


def data_heterogeneous_masked():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    # return the categories
    return X, y, [True, False, True]


def data_heterogeneous_unordered_multiclass():
    rng = np.random.RandomState(42)
    X = np.empty((50, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(50, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(["a", "b", "c"], size=50).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=50)
    y = np.array([0] * 10 + [1] * 15 + [2] * 25)
    # return the categories
    return X, y, [0, 3]


def data_sparse(format):
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=np.float64)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.randint(3, size=30)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    X = sparse.csr_matrix(X) if format == "csr" else sparse.csc_matrix(X)
    return X, y, [0, 3]


@pytest.mark.parametrize(
    'center,surface_point',
    [
        (CENTERS[0], SURFACE_POINTS[0]),
        (CENTERS[1], SURFACE_POINTS[1]),
        (CENTERS[2], SURFACE_POINTS[2]),
        (CENTERS[3], SURFACE_POINTS[3]),
    ],
)
def test_make_geometric_sample_hypersphere(center, surface_point):
    """Test the generation of points inside a hypersphere."""
    point = _make_geometric_sample(center, surface_point, 0.0, 0.0, RANDOM_STATE)
    rel_point = point - center
    rel_surface_point = surface_point - center
    np.testing.assert_array_less(0.0, norm(rel_surface_point) - norm(rel_point))


@pytest.mark.parametrize(
    'surface_point,deformation_factor',
    [
        (np.array([1.0, 0.0]), 0.0),
        (2.6 * np.array([0.0, 1.0]), 0.25),
        (3.2 * np.array([0.0, 1.0, 0.0, 0.0]), 0.50),
        (0.5 * np.array([0.0, 0.0, 1.0]), 0.75),
        (6.7 * np.array([0.0, 0.0, 1.0, 0.0, 0.0]), 1.0),
    ],
)
def test_make_geometric_sample_half_hypersphere(surface_point, deformation_factor):
    """Test the generation of points inside a hypersphere."""
    center = np.zeros(surface_point.shape)
    point = _make_geometric_sample(
        center, surface_point, 1.0, deformation_factor, RANDOM_STATE
    )
    np.testing.assert_array_less(0.0, norm(surface_point) - norm(point))
    np.testing.assert_array_less(0.0, np.dot(point, surface_point))


@pytest.mark.parametrize(
    'center,surface_point,truncation_factor',
    [
        (center, surface_point, truncation_factor)
        for center, surface_point in zip(CENTERS, SURFACE_POINTS)
        for truncation_factor in TRUNCATION_FACTORS
    ],
)
def test_make_geometric_sample_line_segment(center, surface_point, truncation_factor):
    """Test the generation of points on a line segment."""
    point = _make_geometric_sample(
        center, surface_point, truncation_factor, 1.0, RANDOM_STATE
    )
    rel_point = point - center
    rel_surface_point = surface_point - center
    dot_product = np.dot(rel_point, rel_surface_point)
    norms_product = norm(rel_point) * norm(rel_surface_point)
    np.testing.assert_array_less(0.0, norm(rel_surface_point) - norm(rel_point))
    dot_product = (
        np.abs(dot_product) if truncation_factor == 0.0 else (-1) * dot_product
    )
    np.testing.assert_allclose(np.abs(dot_product) / norms_product, 1.0)


def test_gsmote_default_init():
    """Test the intialization with default parameters."""
    gsmote = GeometricSMOTE()
    assert gsmote.sampling_strategy == "auto"
    assert gsmote.random_state is None
    assert gsmote.truncation_factor == 1.0
    assert gsmote.deformation_factor == 0.0
    assert gsmote.selection_strategy == "combined"
    assert gsmote.k_neighbors == 5
    assert gsmote.categorical_features is None
    assert gsmote.n_jobs == 1


def test_gsmote_fit():
    """Test fit method."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(
        random_state=RND_SEED, n_samples=n_samples, weights=weights
    )
    gsmote = GeometricSMOTE(random_state=RANDOM_STATE).fit(X, y)
    assert gsmote.sampling_strategy_ == {1: 40}


def test_gsmote_invalid_selection_strategy():
    """Test invalid selection strategy."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(
        random_state=RND_SEED, n_samples=n_samples, weights=weights
    )
    gsmote = GeometricSMOTE(random_state=RANDOM_STATE, selection_strategy='Minority')
    with pytest.raises(ValueError):
        gsmote.fit_resample(X, y)


@pytest.mark.parametrize('selection_strategy', ['combined', 'minority', 'majority'])
def test_gsmote_nn(selection_strategy):
    """Test nearest neighbors object."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(
        random_state=RND_SEED, n_samples=n_samples, weights=weights
    )
    gsmote = GeometricSMOTE(
        random_state=RANDOM_STATE, selection_strategy=selection_strategy
    )
    _ = gsmote.fit_resample(X, y)
    if selection_strategy in ("minority", "combined"):
        assert gsmote.nns_pos_.n_neighbors == gsmote.k_neighbors + 1
    if selection_strategy in ("majority", "combined"):
        assert gsmote.nn_neg_.n_neighbors == 1


@pytest.mark.parametrize(
    'selection_strategy, truncation_factor, deformation_factor',
    [
        (selection_strategy, truncation_factor, deformation_factor)
        for selection_strategy in SELECTION_STRATEGY
        for truncation_factor in TRUNCATION_FACTORS
        for deformation_factor in DEFORMATION_FACTORS
    ],
)
def test_gsmote_fit_resample_binary(
    selection_strategy, truncation_factor, deformation_factor
):
    """Test fit and sample for binary class case."""
    n_maj, n_min, step, min_coor, max_coor = 12, 5, 0.5, 0.0, 8.5
    X = np.repeat(np.arange(min_coor, max_coor, step), 2).reshape(-1, 2)
    y = np.concatenate([np.repeat(0, n_maj), np.repeat(1, n_min)])
    radius = np.sqrt(0.5) * step
    k_neighbors = 1
    gsmote = GeometricSMOTE(
        "auto",
        RANDOM_STATE,
        truncation_factor,
        deformation_factor,
        selection_strategy,
        k_neighbors,
    )
    X_resampled, y_resampled = gsmote.fit_resample(X, y)
    assert gsmote.sampling_strategy_ == {1: (n_maj - n_min)}
    assert y_resampled.sum() == n_maj
    np.testing.assert_array_less(X[n_maj - 1] - radius, X_resampled[n_maj + n_min])


@pytest.mark.parametrize(
    'selection_strategy, truncation_factor, deformation_factor',
    [
        (selection_strategy, truncation_factor, deformation_factor)
        for selection_strategy in SELECTION_STRATEGY
        for truncation_factor in TRUNCATION_FACTORS
        for deformation_factor in DEFORMATION_FACTORS
    ],
)
def test_gsmote_fit_resample_multiclass(
    selection_strategy, truncation_factor, deformation_factor
):
    """Test fit and sample for multiclass case."""
    n_samples, weights = 100, [0.75, 0.15, 0.10]
    X, y = make_classification(
        random_state=RND_SEED,
        n_samples=n_samples,
        weights=weights,
        n_classes=3,
        n_informative=5,
    )
    k_neighbors, majority_label = 1, 0
    gsmote = GeometricSMOTE(
        "auto",
        RANDOM_STATE,
        truncation_factor,
        deformation_factor,
        selection_strategy,
        k_neighbors,
    )
    _, y_resampled = gsmote.fit_resample(X, y)
    assert majority_label not in gsmote.sampling_strategy_.keys()
    np.testing.assert_array_equal(np.unique(y), np.unique(y_resampled))
    assert len(set(Counter(y_resampled).values())) == 1


def test_categorical_error():
    X, y, _ = data_heterogeneous_unordered()
    categorical_features = [0, 10]
    gsmote = GeometricSMOTE(random_state=0, categorical_features=categorical_features)
    with pytest.raises(ValueError, match="indices are out of range"):
        gsmote.fit_resample(X, y)


@pytest.mark.parametrize(
    "data",
    [
        data_heterogeneous_ordered(),
        data_heterogeneous_unordered(),
        data_heterogeneous_masked(),
        data_sparse("csr"),
        data_sparse("csc"),
    ],
)
def test_gsmotenc(data):
    X, y, categorical_features = data
    gsmote = GeometricSMOTE(random_state=0, categorical_features=categorical_features)
    X_resampled, y_resampled = gsmote.fit_resample(X, y)

    assert X_resampled.dtype == X.dtype

    categorical_features = np.array(categorical_features)
    if categorical_features.dtype == bool:
        categorical_features = np.flatnonzero(categorical_features)
    for cat_idx in categorical_features:
        if sparse.issparse(X):
            assert set(X[:, cat_idx].data) == set(X_resampled[:, cat_idx].data)
            assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype
        else:
            assert set(X[:, cat_idx]) == set(X_resampled[:, cat_idx])
            assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype


# part of the common test which apply to GSMOTE-NC even if it is not default
# constructible
def test_smotenc_check_target_type():
    X, _, categorical_features = data_heterogeneous_unordered()
    y = np.linspace(0, 1, 30)
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=0)
    with pytest.raises(ValueError, match="Unknown label type: 'continuous'"):
        gsmote.fit_resample(X, y)
    rng = np.random.RandomState(42)
    y = rng.randint(2, size=(20, 3))
    msg = "Multilabel and multioutput targets are not supported."
    with pytest.raises(ValueError, match=msg):
        gsmote.fit_resample(X, y)


def test_smotenc_samplers_one_label():
    X, _, categorical_features = data_heterogeneous_unordered()
    y = np.zeros(30)
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=0)
    with pytest.raises(ValueError, match="needs to have more than 1 class"):
        gsmote.fit(X, y)


def test_smotenc_fit():
    X, y, categorical_features = data_heterogeneous_unordered()
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=0)
    gsmote.fit_resample(X, y)
    assert hasattr(
        gsmote, "sampling_strategy_"
    ), "No fitted attribute sampling_strategy_"


def test_smotenc_fit_resample():
    X, y, categorical_features = data_heterogeneous_unordered()
    target_stats = Counter(y)
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=0)
    _, y_res = gsmote.fit_resample(X, y)
    _ = Counter(y_res)
    n_samples = max(target_stats.values())
    assert all(value >= n_samples for value in Counter(y_res).values())


def test_smotenc_fit_resample_sampling_strategy():
    X, y, categorical_features = data_heterogeneous_unordered_multiclass()
    expected_stat = Counter(y)[1]
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=0)
    sampling_strategy = {2: 25, 0: 25}
    gsmote.set_params(sampling_strategy=sampling_strategy)
    X_res, y_res = gsmote.fit_resample(X, y)
    assert Counter(y_res)[1] == expected_stat


def test_smotenc_pandas():
    pd = pytest.importorskip("pandas")
    # Check that the samplers handle pandas dataframe and pandas series
    X, y, categorical_features = data_heterogeneous_unordered_multiclass()
    X_pd = pd.DataFrame(X)
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=0)
    X_res_pd, y_res_pd = gsmote.fit_resample(X_pd, y)
    X_res, y_res = gsmote.fit_resample(X, y)
    assert_array_equal(X_res_pd.to_numpy(), X_res)
    assert_allclose(y_res_pd, y_res)


def test_smotenc_preserve_dtype():
    X, y = make_classification(
        n_samples=50,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    # Cast X and y to not default dtype
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    gsmote = GeometricSMOTE(categorical_features=[1], random_state=0)
    X_res, y_res = gsmote.fit_resample(X, y)
    assert X.dtype == X_res.dtype, "X dtype is not preserved"
    assert y.dtype == y_res.dtype, "y dtype is not preserved"


@pytest.mark.parametrize("categorical_features", [[True, True, True], [0, 1, 2]])
def test_smotenc_raising_error_all_categorical(categorical_features):
    X, y = make_classification(
        n_features=3,
        n_informative=1,
        n_redundant=1,
        n_repeated=0,
        n_clusters_per_class=1,
    )
    gsmote = GeometricSMOTE(categorical_features=categorical_features)
    err_msg = "GeometricSMOTE is not designed to work only with categorical features"
    with pytest.raises(ValueError, match=err_msg):
        gsmote.fit_resample(X, y)


def test_smote_nc_with_null_median_std():
    # Non-regression test for #662
    # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/662
    data = np.array(
        [
            [1, 2, 1, 'A'],
            [2, 1, 2, 'A'],
            [1, 2, 3, 'B'],
            [1, 2, 4, 'C'],
            [1, 2, 5, 'C'],
        ],
        dtype="object",
    )
    labels = np.array(
        ['class_1', 'class_1', 'class_1', 'class_2', 'class_2'], dtype=object
    )
    gsmote = GeometricSMOTE(
        categorical_features=[3],
        k_neighbors=1,
        selection_strategy='minority',
        random_state=0,
    )
    X_res, y_res = gsmote.fit_resample(data, labels)
    # check that the categorical feature is not random but correspond to the
    # categories seen in the minority class samples
    assert X_res[-1, -1] == 'C'
