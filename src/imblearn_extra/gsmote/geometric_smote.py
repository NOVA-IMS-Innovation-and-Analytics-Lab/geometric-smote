"""Class to perform over-sampling using Geometric SMOTE."""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca    <jpmrfonseca@gmail.com>
# License: BSD 3 clause

import math
from collections import Counter
from collections.abc import Callable

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution, check_neighbors_object, check_target_type
from imblearn.utils._docstring import _random_state_docstring
from numpy.linalg import norm
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array, check_random_state
from typing_extensions import Self

SELECTION_STRATEGIES = ('combined', 'majority', 'minority')


def make_geometric_sample(
    center: NDArray,
    surface_point: NDArray,
    truncation_factor: float,
    deformation_factor: float,
    random_state: np.random.RandomState,
) -> NDArray:
    """A support function that returns an artificial point.

    Args:
        center:
            The center point.

        surface_point:
            The point on the surface of the hypersphere.

        truncation_factor:
            The truncation factor of the algorithm.

        deformation_factor:
            The defirmation factor of the algorithm.

        random_state:
            The random state of the process.

    Returns:
        geometric_sample:
            The generated geometric sample.
    """

    # Zero radius case
    if np.array_equal(center, surface_point):
        return center

    # Generate a point on the surface of a unit hyper-sphere
    radius = norm(center - surface_point)
    normal_samples = random_state.normal(size=center.size)
    point_on_unit_sphere = normal_samples / norm(normal_samples)
    point: NDArray = (random_state.uniform(size=1) ** (1 / center.size)) * point_on_unit_sphere

    # Parallel unit vector
    parallel_unit_vector = (surface_point - center) / norm(surface_point - center)

    # Truncation
    close_to_opposite_boundary = truncation_factor > 0 and np.dot(point, parallel_unit_vector) < truncation_factor - 1
    close_to_boundary = truncation_factor < 0 and np.dot(point, parallel_unit_vector) > truncation_factor + 1
    if close_to_opposite_boundary or close_to_boundary:
        point -= 2 * np.dot(point, parallel_unit_vector) * parallel_unit_vector

    # Deformation
    parallel_point_position = np.dot(point, parallel_unit_vector) * parallel_unit_vector
    perpendicular_point_position = point - parallel_point_position
    point = parallel_point_position + (1 - deformation_factor) * perpendicular_point_position

    # Translation
    point = center + radius * point

    return point


def populate_categorical_features(
    X_new: NDArray,
    neighbors: NDArray,
    categories_size: list[int] | None,
    random_state: np.random.RandomState,
) -> NDArray:
    """A support function that populates categorical features."""
    if categories_size is not None:
        for start_idx, end_idx in zip(
            np.cumsum(categories_size)[:-1],
            np.cumsum(categories_size)[1:],
            strict=False,
        ):
            col_maxs = neighbors[:, start_idx:end_idx].sum(axis=0)
            is_max = np.isclose(col_maxs, col_maxs.max(axis=0))
            max_idxs = random_state.permutation(np.argwhere(is_max))
            col_sels = max_idxs[0]
            ys = start_idx + col_sels
            X_new[start_idx:end_idx] = 0
            X_new[ys] = 1
    return X_new


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class GeometricSMOTE(BaseOverSampler):
    """Class to to perform over-sampling using Geometric SMOTE.

    This algorithm is an implementation of Geometric SMOTE, a geometrically
    enhanced drop-in replacement for SMOTE. Read more in the [user_guide].

    Args:
        categorical_features:
            Specified which features are categorical. Can either be:

                - array of indices specifying the categorical features.

                - mask array of shape (n_features, ) and `bool` dtype for which
                `True` indicates the categorical features.

        sampling_strategy:
            Sampling information to resample the data set.

            - When `float`, it corresponds to the desired ratio of the number of
            samples in the minority class over the number of samples in the
            majority class after resampling. It is only available for binary
            classification.

            - When `str`, specify the class targeted by the resampling. The
            number of samples in the different classes will be equalized.
            Possible choices are:
                - `'minority'`: resample only the minority class.
                - `'not minority'`: resample all classes but the minority class.
                - `'not majority'`: resample all classes but the majority class.
                - `'all'`: resample all classes.
                - `'auto'`: equivalent to `'not majority'`.

            - When `dict`, the keys correspond to the targeted classes. The
            values correspond to the desired number of samples for each targeted
            class.

            - When callable, function taking `y` and returns a `dict`. The keys
            correspond to the targeted classes. The values correspond to the
            desired number of samples for each class.

        random_state:
            Control the randomization of the algorithm.

            - If `int`, it is the seed used by the random number
            generator.
            - If `np.random.RandomState` instance, it is the random number
            generator.
            - If `None`, the random number generator is the `RandomState`
            instance used by `np.random`.

        truncation_factor:
            The type of truncation. The values should be in the [-1.0, 1.0] range.

        deformation_factor:
            The type of geometry. The values should be in the [0.0, 1.0] range.

        selection_strategy:
            The type of Geometric SMOTE algorithm with the following options:
            `'combined'`, `'majority'`, `'minority'`.

        k_neighbors:
            If `int`, number of nearest neighbours to use when synthetic
            samples are constructed for the minority method.  If object, an estimator
            that inherits from `sklearn.neighbors.base.KNeighborsMixin` class that
            will be used to find the k_neighbors.

        n_jobs:
            The number of threads to open if possible.

    Attributes:
        n_features_in_ : int
            Number of features in the input dataset.

        nns_pos_ : estimator object
            Validated k-nearest neighbours created from the `k_neighbors` parameter. It is
            used to find the nearest neighbors of the same class of a selected
            observation.

        nn_neg_ : estimator object
            Validated k-nearest neighbours created from the `k_neighbors` parameter. It is
            used to find the nearest neighbor of the remaining classes (k=1) of a selected
            observation.

        random_state_ (np.random.RandomState):
            An instance of `np.random.RandomState` class.

        sampling_strategy_ (dict[int, int]):
            Actual sampling strategy.

    Examples:
        >>> from collections import Counter
        >>> from sklearn.datasets import make_classification
        >>> from gsmote import GeometricSMOTE # doctest: +NORMALIZE_WHITESPACE
        >>> X, y = make_classification(n_classes=2, class_sep=2,
        ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
        ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
        >>> print('Original dataset shape %s' % Counter(y))
        Original dataset shape Counter({{1: 900, 0: 100}})
        >>> gsmote = GeometricSMOTE(random_state=1)
        >>> X_resampled, y_resampled = gsmote.fit_resample(X, y)
        >>> print('Resampled dataset shape %s' % Counter(y_resampled))
        Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    def __init__(
        self: Self,
        sampling_strategy: dict[int, int] | str | float | Callable = 'auto',
        k_neighbors: NearestNeighbors | int = 5,
        truncation_factor: float = 1.0,
        deformation_factor: float = 0.0,
        selection_strategy: str = 'combined',
        categorical_features: ArrayLike | None = None,
        random_state: np.random.RandomState | int | None = None,
        n_jobs: int | None = 1,
    ) -> None:
        """Initialize oversampler."""
        super().__init__(sampling_strategy=sampling_strategy)
        self.k_neighbors = k_neighbors
        self.truncation_factor = truncation_factor
        self.deformation_factor = deformation_factor
        self.selection_strategy = selection_strategy
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _validate_estimators(self: Self, X: NDArray) -> Self:
        """Validate nearest neighbors estimators."""

        # Check random state
        self.random_state_ = check_random_state(self.random_state)

        # Validate strategy
        if self.selection_strategy not in SELECTION_STRATEGIES:
            error_msg = (
                'Unknown selection_strategy for Geometric SMOTE algorithm. '
                f'Choices are {SELECTION_STRATEGIES}. Got {self.selection_strategy} instead.'
            )
            raise ValueError(error_msg)

        # Create nearest neighbors object for positive class
        if self.selection_strategy in ('minority', 'combined'):
            self.nns_pos_ = check_neighbors_object(
                'nns_positive',
                self.k_neighbors,
                additional_neighbor=1,
            )
            self.nns_pos_.set_params(n_jobs=self.n_jobs)

        # Create nearest neighbors object for negative class
        if self.selection_strategy in ('majority', 'combined'):
            self.nn_neg_ = check_neighbors_object('nn_negative', nn_object=1)
            self.nn_neg_.set_params(n_jobs=self.n_jobs)

        # Create one hot encoder object
        if self.categorical_features is not None:
            self.ohe_ = OneHotEncoder(
                sparse_output=True,
                handle_unknown='ignore',
                dtype=X.dtype if X.dtype.name != 'object' else np.float64,
            )
            self.ohe_.fit(X[:, self.categorical_features_])

        return self

    def _validate_categorical_features(self: Self) -> Self:
        """Validate categorical features."""

        if self.categorical_features is None:
            self.categorical_features_ = np.flatnonzero([])
            self.continuous_features_ = np.arange(self.n_features_in_)
            return self

        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == 'bool':
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any(index not in np.arange(self.n_features_in_) for index in categorical_features):
                error_msg = (
                    'Some of the categorical indices are out of range. Indices'
                    f' should be between 0 and {self.n_features_in_ - 1}.'
                )
                raise ValueError(error_msg)
            self.categorical_features_ = np.sort(categorical_features)
        self.continuous_features_ = np.setdiff1d(
            np.arange(self.n_features_in_),
            self.categorical_features_,
        )

        if self.categorical_features_.size == self.n_features_in_:
            error_msg = (
                'GeometricSMOTE is not designed to work only with categorical '
                'features. It requires some numerical features.'
            )
            raise ValueError(error_msg)

        return self

    def _check_X_y(  # noqa: N802
        self: Self,
        X: ArrayLike | sparse.csc_matrix | sparse.csr_matrix,
        y: ArrayLike,
    ) -> tuple[NDArray, NDArray, bool]:
        """Check input and output data."""
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        validated_data: tuple[NDArray | sparse.csc_matrix | sparse.csr_matrix, NDArray] = self._validate_data(
            X,
            y,
            reset=True,
            dtype=None,
            accept_sparse=['csr', 'csc'],
        )
        X, y = validated_data
        if not isinstance(X, np.ndarray):
            X = X.toarray()
        return X, y, binarize_y

    def _make_geometric_samples(  # noqa: C901
        self: Self,
        X_init: NDArray,
        X: NDArray,
        y: NDArray,
        pos_class_label: str | int,
        n_samples: int,
    ) -> tuple[NDArray, NDArray, list[NDArray]]:
        """A support function that returns artificials samples."""

        # Return zero new samples
        if n_samples == 0:
            return (
                np.array([], dtype=X.dtype).reshape(0, X.shape[1]),
                np.array([], dtype=y.dtype),
                [],
            )

        # Select positive class samples
        X_pos = X[y == pos_class_label]

        # Force minority strategy if no negative class samples are present
        self.selection_strategy_ = 'minority' if X.shape[0] == X_pos.shape[0] else self.selection_strategy

        # Minority or combined strategy
        if self.selection_strategy_ in ('minority', 'combined'):
            self.nns_pos_.fit(X_pos)
            points_pos = self.nns_pos_.kneighbors(X_pos)[1][:, 1:]
            samples_indices = self.random_state_.randint(
                low=0,
                high=len(points_pos.flatten()),
                size=n_samples,
            )
            rows = np.floor_divide(samples_indices, points_pos.shape[1])
            cols = np.mod(samples_indices, points_pos.shape[1])

        # Majority or combined strategy
        if self.selection_strategy_ in ('majority', 'combined'):
            X_neg = X[y != pos_class_label]
            self.nn_neg_.fit(X_neg)
            points_neg = self.nn_neg_.kneighbors(X_pos)[1]
            if self.selection_strategy_ == 'majority':
                samples_indices = self.random_state_.randint(
                    low=0,
                    high=len(points_neg.flatten()),
                    size=n_samples,
                )
                rows = np.floor_divide(samples_indices, points_neg.shape[1])
                cols = np.mod(samples_indices, points_neg.shape[1])

        # Case that the median std equals to zeros
        if self.categorical_features is not None and math.isclose(self.median_std_, 0):
            X[:, self.continuous_features_.size :] = self.ohe_.transform(
                X_init[:, self.categorical_features_],
            ).toarray()
            X_pos = X[y == pos_class_label]
            if self.selection_strategy_ in ('majority', 'combined'):
                X_neg = X[y != pos_class_label]

        # Generate new samples
        X_new = np.zeros((n_samples, X.shape[1]))
        all_neighbors = []
        for ind, (row, col) in enumerate(zip(rows, cols, strict=False)):
            # Define center point
            center = X_pos[row]

            # Minority strategy
            if self.selection_strategy_ == 'minority':
                surface_point = X_pos[points_pos[row, col]]
                neighbors = X_pos[points_pos[row]]

            # Majority strategy
            elif self.selection_strategy_ == 'majority':
                surface_point = X_neg[points_neg[row, col]]
                neighbors = X_neg[points_neg[row]]

            # Combined strategy
            else:
                surface_point_pos = X_pos[points_pos[row, col]]
                surface_point_neg = X_neg[points_neg[row, 0]]
                radius_pos = norm(center - surface_point_pos)
                radius_neg = norm(center - surface_point_neg)
                surface_point = surface_point_neg if radius_pos > radius_neg else surface_point_pos
                neighbors = np.vstack([X_pos[points_pos[row]], X_neg[points_neg[row]]])

            if self.categorical_features is not None:
                all_neighbors.append(neighbors)

            # Append new sample - no categorical features
            X_new[ind] = make_geometric_sample(
                center,
                surface_point,
                self.truncation_factor,
                self.deformation_factor,
                self.random_state_,
            )

        # Create new samples for target variable
        y_new = np.array([pos_class_label] * len(samples_indices))

        return X_new, y_new, all_neighbors

    def _populate_categorical_features(
        self: Self,
        X_new: NDArray,
        y_new: NDArray,
        all_neighbors: list[NDArray],
    ) -> tuple[NDArray, NDArray]:
        """A support function that populates categorical features."""
        categories_size = (
            [self.continuous_features_.size] + [cat.size for cat in self.ohe_.categories_]
            if self.categorical_features is not None
            else None
        )
        for ind, neighbors in enumerate(all_neighbors):
            X_new[ind] = populate_categorical_features(
                X_new[ind],
                neighbors,
                categories_size,
                self.random_state_,
            )
        return X_new, y_new

    def _encode_categorical(self: Self, X: NDArray, y: NDArray) -> NDArray:
        """Encode categorical features."""

        if self.categorical_features is None:
            return X

        # Compute the median of the standard deviation of the minority class
        class_minority = Counter(y).most_common()[-1][0]

        # Calcuate variance
        X_continuous = check_array(X[:, self.continuous_features_], dtype=np.float64)
        X_minority_continuous = X_continuous[np.flatnonzero(y == class_minority)]
        var = X_minority_continuous.var(axis=0)
        self.median_std_ = np.median(np.sqrt(var))

        # OneHotEncoder
        X_categorical = X[:, self.categorical_features_]
        X_ohe_categorical = self.ohe_.transform(X_categorical)
        X_ohe_categorical.data = (
            np.ones_like(X_ohe_categorical.data, dtype=X_ohe_categorical.dtype) * self.median_std_ / 2
        )
        X_encoded = np.hstack([X_continuous, X_ohe_categorical.toarray()])

        return X_encoded

    def _decode_categorical(self: Self, X_init: NDArray, X_resampled: NDArray) -> NDArray:
        """Reverses the encoding of the categorical features."""

        if self.categorical_features is None:
            return X_resampled.astype(X_init.dtype)

        if math.isclose(self.median_std_, 0):
            X_resampled[: X_init.shape[0], self.continuous_features_.size :] = self.ohe_.transform(
                X_init[:, self.categorical_features_],
            ).toarray()

        indices_reordered = np.argsort(
            np.hstack((self.continuous_features_, self.categorical_features_)),
        )
        X_resampled = np.hstack(
            (
                X_resampled[:, : self.continuous_features_.size],
                self.ohe_.inverse_transform(X_resampled[:, self.continuous_features_.size :]),
            ),
        )[:, indices_reordered].astype(X_init.dtype)
        return X_resampled

    def _fit_resample(
        self: Self,
        X: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray]:
        # Validation
        self._validate_categorical_features()._validate_estimators(X)

        # Preprocess categorical data
        X_init = X.copy()
        X = self._encode_categorical(X, y)

        # Copy data
        X_resampled, y_resampled = X.copy(), y.copy()

        # Resample
        for class_label, n_samples in self.sampling_strategy_.items():
            # Apply G-SMOTE mechanism
            X_new, y_new, all_neighbors = self._make_geometric_samples(
                X_init,
                X,
                y,
                class_label,
                n_samples,
            )

            # Apply SMOTE-NC mechanism
            X_new, y_new = self._populate_categorical_features(
                X_new,
                y_new,
                all_neighbors,
            )

            # Append new data
            X_resampled, y_resampled = (
                np.vstack((X_resampled, X_new)),
                np.hstack((y_resampled, y_new)),
            )

        # Reverse the encoding of the categorical features
        X_resampled = self._decode_categorical(X_init, X_resampled)

        return X_resampled, y_resampled
