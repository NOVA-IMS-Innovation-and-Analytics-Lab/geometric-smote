"""Test the geometric_smote module."""

from collections import Counter

import numpy as np
import pytest
from imblearn_extra.gsmote import SELECTION_STRATEGIES, GeometricSMOTE, make_geometric_sample
from numpy.linalg import norm
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

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


def create_heterogeneous_ordered_data_indices():
    """Create heterogeneous data with ordered features."""
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    X[:, :2] = rng.randn(30, 2)
    X[:, 2] = rng.choice(['a', 'b', 'c'], size=30).astype(object)
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    return X, y, [2, 3]


def create_heterogeneous_unordered_data_indices():
    """Create heterogeneous data with unordered features."""
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    X[:, [1, 2]] = rng.randn(30, 2)
    X[:, 0] = rng.choice(['a', 'b', 'c'], size=30).astype(object)
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    return X, y, [0, 3]


def create_heterogeneous_unordered_data_mask():
    """Create heterogeneous data with unordered features."""
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    X[:, [1, 2]] = rng.randn(30, 2)
    X[:, 0] = rng.choice(['a', 'b', 'c'], size=30).astype(object)
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    return X, y, [True, False, True]


def create_heterogeneous_unordered_multiclass_data_indices():
    """Create heterogeneous multiclass data with unordered features."""
    rng = np.random.RandomState(42)
    X = np.empty((50, 4), dtype=object)
    X[:, [1, 2]] = rng.randn(50, 2)
    X[:, 0] = rng.choice(['a', 'b', 'c'], size=50).astype(object)
    X[:, 3] = rng.randint(3, size=50)
    y = np.array([0] * 10 + [1] * 15 + [2] * 25)
    return X, y, [0, 3]


def create_sparse_data(fmt: str):
    """Create sparse data."""
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=np.float64)
    X[:, [1, 2]] = rng.randn(30, 2)
    X[:, 0] = rng.randint(3, size=30)
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    X = sparse.csr_matrix(X) if fmt == 'csr' else sparse.csc_matrix(X)
    return X, y, [0, 3]


@pytest.mark.parametrize(
    ('center', 'surface_point'),
    [
        (CENTERS[0], SURFACE_POINTS[0]),
        (CENTERS[1], SURFACE_POINTS[1]),
        (CENTERS[2], SURFACE_POINTS[2]),
        (CENTERS[3], SURFACE_POINTS[3]),
    ],
)
def test_make_geometric_sample_hypersphere(center, surface_point):
    """Test the generation of points inside a hypersphere."""
    point = make_geometric_sample(center, surface_point, 0.0, 0.0, RANDOM_STATE)
    rel_point = point - center
    rel_surface_point = surface_point - center
    np.testing.assert_array_less(0.0, norm(rel_surface_point) - norm(rel_point))


@pytest.mark.parametrize(
    ('surface_point', 'deformation_factor'),
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
    point = make_geometric_sample(center, surface_point, 1.0, deformation_factor, RANDOM_STATE)
    np.testing.assert_array_less(0.0, norm(surface_point) - norm(point))
    np.testing.assert_array_less(0.0, np.dot(point, surface_point))


@pytest.mark.parametrize(
    ('center', 'surface_point', 'truncation_factor'),
    [
        (center, surface_point, truncation_factor)
        for center, surface_point in zip(CENTERS, SURFACE_POINTS, strict=False)
        for truncation_factor in TRUNCATION_FACTORS
    ],
)
def test_make_geometric_sample_line_segment(center, surface_point, truncation_factor):
    """Test the generation of points on a line segment."""
    point = make_geometric_sample(center, surface_point, truncation_factor, 1.0, RANDOM_STATE)
    rel_point = point - center
    rel_surface_point = surface_point - center
    dot_product = np.dot(rel_point, rel_surface_point)
    norms_product = norm(rel_point) * norm(rel_surface_point)
    np.testing.assert_array_less(0.0, norm(rel_surface_point) - norm(rel_point))
    truncation_factor_min = 0.0
    dot_product = np.abs(dot_product) if truncation_factor == truncation_factor_min else (-1) * dot_product
    np.testing.assert_allclose(np.abs(dot_product) / norms_product, 1.0)


def test_default_init():
    """Test the intialization with default parameters."""
    gsmote = GeometricSMOTE()
    truncation_factor_default = 1.0
    deformation_factor_min = 0.0
    k_neighbors = 5
    assert gsmote.sampling_strategy == 'auto'
    assert gsmote.random_state is None
    assert gsmote.truncation_factor == truncation_factor_default
    assert gsmote.deformation_factor == deformation_factor_min
    assert gsmote.selection_strategy == "combined"
    assert gsmote.k_neighbors == k_neighbors
    assert gsmote.categorical_features is None
    assert gsmote.n_jobs == 1


def test_fit():
    """Test fit method."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(random_state=RND_SEED, n_samples=n_samples, weights=weights)
    gsmote = GeometricSMOTE(random_state=RANDOM_STATE).fit(X, y)
    assert gsmote.sampling_strategy_ == {1: 40}


def test_error_wrong_selection_strategy():
    """Test wrong selection strategy."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(random_state=RND_SEED, n_samples=n_samples, weights=weights)
    gsmote = GeometricSMOTE(random_state=RANDOM_STATE, selection_strategy='Minority')
    with pytest.raises(ValueError, match='Unknown selection_strategy for Geometric SMOTE'):
        gsmote.fit_resample(X, y)


@pytest.mark.parametrize('selection_strategy', ['combined', 'minority', 'majority'])
def test_nearest_neighbors(selection_strategy):
    """Test nearest neighbors object."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(random_state=RND_SEED, n_samples=n_samples, weights=weights)
    gsmote = GeometricSMOTE(random_state=RANDOM_STATE, selection_strategy=selection_strategy)
    _ = gsmote.fit_resample(X, y)
    if selection_strategy in ("minority", "combined"):
        assert gsmote.nns_pos_.n_neighbors == gsmote.k_neighbors + 1
    if selection_strategy in ("majority", "combined"):
        assert gsmote.nn_neg_.n_neighbors == 1


@pytest.mark.parametrize(
    ('selection_strategy', 'truncation_factor', 'deformation_factor'),
    [
        (selection_strategy, truncation_factor, deformation_factor)
        for selection_strategy in set(SELECTION_STRATEGIES).difference(['majority'])
        for truncation_factor in TRUNCATION_FACTORS
        for deformation_factor in DEFORMATION_FACTORS
    ],
)
def test_binary_fit_resample(selection_strategy, truncation_factor, deformation_factor):
    """Test fit and sample for binary class case."""
    n_maj, n_min, step, min_coor, max_coor = 12, 5, 0.5, 0.0, 8.5
    X = np.repeat(np.arange(min_coor, max_coor, step), 2).reshape(-1, 2)
    y = np.concatenate([np.repeat(0, n_maj), np.repeat(1, n_min)])
    radius = np.sqrt(0.5) * step
    k_neighbors = 1
    gsmote = GeometricSMOTE(
        'auto',
        k_neighbors,
        truncation_factor,
        deformation_factor,
        selection_strategy,
        random_state=RANDOM_STATE,
    )
    X_resampled, y_resampled = gsmote.fit_resample(X, y)
    assert gsmote.sampling_strategy_ == {1: (n_maj - n_min)}
    assert y_resampled.sum() == n_maj
    np.testing.assert_array_less(X[n_maj - 1] - radius, X_resampled[n_maj + n_min])


@pytest.mark.parametrize(
    ('selection_strategy', 'truncation_factor', 'deformation_factor'),
    [
        (selection_strategy, truncation_factor, deformation_factor)
        for selection_strategy in SELECTION_STRATEGIES
        for truncation_factor in TRUNCATION_FACTORS
        for deformation_factor in DEFORMATION_FACTORS
    ],
)
def test_multiclass_fit_resample(selection_strategy, truncation_factor, deformation_factor):
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
        'auto',
        k_neighbors,
        truncation_factor,
        deformation_factor,
        selection_strategy,
        random_state=RANDOM_STATE,
    )
    _, y_resampled = gsmote.fit_resample(X, y)
    assert majority_label not in gsmote.sampling_strategy_
    np.testing.assert_array_equal(np.unique(y), np.unique(y_resampled))
    assert len(set(Counter(y_resampled).values())) == 1


def test_wrong_categorical_indices():
    """Test error for indices out of range."""
    X, y, _ = create_heterogeneous_ordered_data_indices()
    categorical_features = [0, 10]
    gsmote = GeometricSMOTE(random_state=0, categorical_features=categorical_features)
    with pytest.raises(ValueError, match="indices are out of range"):
        gsmote.fit_resample(X, y)


@pytest.mark.parametrize(
    'data',
    [
        create_heterogeneous_ordered_data_indices(),
        create_heterogeneous_unordered_data_indices(),
        create_heterogeneous_unordered_data_mask(),
        create_sparse_data('csr'),
        create_sparse_data('csc'),
    ],
)
def test_categorical_features(data):
    """Test categorical features for various types of data."""
    X, y, categorical_features = data
    gsmote = GeometricSMOTE(random_state=0, categorical_features=categorical_features)
    categorical_features = np.array(categorical_features)
    if categorical_features.dtype == bool:
        categorical_features = np.flatnonzero(categorical_features)
    X_resampled, _ = gsmote.fit_resample(X, y)
    if sparse.issparse(X):
        X = X.toarray()
    assert X_resampled.dtype == X.dtype
    for cat_idx in categorical_features:
        np.testing.assert_array_equal(X[: X.shape[0], cat_idx], X_resampled[: X.shape[0], cat_idx])
        assert set(X[:, cat_idx]) == set(X_resampled[:, cat_idx])
        assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype


def test_wrong_target_type():
    """Test error for wrong target type."""
    X, _, categorical_features = create_heterogeneous_unordered_data_indices()
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=RANDOM_STATE)
    with pytest.raises(ValueError, match='Unknown label type: continuous'):
        gsmote.fit_resample(X, y=np.linspace(0, 1, 30))
    with pytest.raises(ValueError, match='Multilabel and multioutput targets are not supported.'):
        gsmote.fit_resample(X, RANDOM_STATE.randint(2, size=(20, 3)))


def test_error_one_label():
    """Test error for one label."""
    X, _, categorical_features = create_heterogeneous_unordered_data_indices()
    y = np.zeros(30)
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=0)
    with pytest.raises(ValueError, match='needs to have more than 1 class'):
        gsmote.fit(X, y)


def test_pandas_fit_resample():
    """Tests fit and resample for pandas input."""
    pd = pytest.importorskip('pandas')
    X, y, categorical_features = create_heterogeneous_unordered_data_indices()
    gsmote = GeometricSMOTE(categorical_features=categorical_features, random_state=0)
    X_res, y_res = gsmote.fit_resample(X, y)
    X_pd = pd.DataFrame(X)
    X_res_pd, y_res_pd = gsmote.fit_resample(X_pd, y)
    np.testing.assert_array_equal(X_res_pd.to_numpy(), X_res)
    np.testing.assert_array_equal(y_res_pd, y_res)


def test_data_types_fit_resample():
    """Tests fit and resample preserves data type."""
    X, y = make_classification(
        n_samples=50,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    gsmote = GeometricSMOTE(categorical_features=[1], random_state=0)
    X_res, y_res = gsmote.fit_resample(X, y)
    assert X.dtype == X_res.dtype
    assert y.dtype == y_res.dtype


@pytest.mark.parametrize("categorical_features", [[True, True, True], [0, 1, 2]])
def test_all_categorical_fit_resample(categorical_features):
    """Tests fit and resample error all categorical features."""
    X, y = make_classification(
        n_features=3,
        n_informative=1,
        n_redundant=1,
        n_repeated=0,
        n_clusters_per_class=1,
    )
    gsmote = GeometricSMOTE(categorical_features=categorical_features)
    err_msg = 'GeometricSMOTE is not designed to work only with categorical features'
    with pytest.raises(ValueError, match=err_msg):
        gsmote.fit_resample(X, y)


def test_null_median_std_fit_resample():
    """Tests fit and resample categorical features null median."""
    X = np.array(
        [
            [1, 2, 1, 'A'],
            [2, 1, 2, 'A'],
            [1, 2, 3, 'B'],
            [1, 2, 4, 'C'],
            [1, 2, 5, 'C'],
        ],
        dtype='object',
    )
    y = np.array(['class_1', 'class_1', 'class_1', 'class_2', 'class_2'], dtype=object)
    gsmote = GeometricSMOTE(
        categorical_features=[3],
        k_neighbors=1,
        selection_strategy='minority',
        random_state=0,
    )
    X_res, _ = gsmote.fit_resample(X, y)
    assert X_res[-1, -1] == 'C'
