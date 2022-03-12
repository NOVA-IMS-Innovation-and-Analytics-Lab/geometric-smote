"""Class to perform over-sampling using Geometric SMOTE."""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca    <jpmrfonseca@gmail.com>
# License: BSD 3 clause

import math
import numpy as np
from collections import Counter
from numpy.linalg import norm
from scipy import sparse
from sklearn.utils import check_random_state, check_array
from sklearn.utils.sparsefuncs_fast import (
    csr_mean_variance_axis0,
    csc_mean_variance_axis0,
)
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_neighbors_object, Substitution, check_target_type
from imblearn.utils._docstring import _random_state_docstring

SELECTION_STRATEGY = ("combined", "majority", "minority")


def _make_geometric_sample(
    center, surface_point, truncation_factor, deformation_factor, random_state
):
    """A support function that returns an artificial point inside
    the geometric region defined by the center and surface points.

    Parameters
    ----------
    center : ndarray, shape (n_features, )
        Center point of the geometric region.

    surface_point : ndarray, shape (n_features, )
        Surface point of the geometric region.

    truncation_factor : float, optional (default=0.0)
        The type of truncation. The values should be in the [-1.0, 1.0] range.

    deformation_factor : float, optional (default=0.0)
        The type of geometry. The values should be in the [0.0, 1.0] range.

    random_state : int, RandomState instance or None
        Control the randomization of the algorithm.

    Returns
    -------
    point : ndarray, shape (n_features, )
            Synthetically generated sample.

    """

    # Zero radius case
    if np.array_equal(center, surface_point):
        return center

    # Generate a point on the surface of a unit hyper-sphere
    radius = norm(center - surface_point)
    normal_samples = random_state.normal(size=center.size)
    point_on_unit_sphere = normal_samples / norm(normal_samples)
    point = (random_state.uniform(size=1) ** (1 / center.size)) * point_on_unit_sphere

    # Parallel unit vector
    parallel_unit_vector = (surface_point - center) / norm(surface_point - center)

    # Truncation
    close_to_opposite_boundary = (
        truncation_factor > 0
        and np.dot(point, parallel_unit_vector) < truncation_factor - 1
    )
    close_to_boundary = (
        truncation_factor < 0
        and np.dot(point, parallel_unit_vector) > truncation_factor + 1
    )
    if close_to_opposite_boundary or close_to_boundary:
        point -= 2 * np.dot(point, parallel_unit_vector) * parallel_unit_vector

    # Deformation
    parallel_point_position = np.dot(point, parallel_unit_vector) * parallel_unit_vector
    perpendicular_point_position = point - parallel_point_position
    point = (
        parallel_point_position
        + (1 - deformation_factor) * perpendicular_point_position
    )

    # Translation
    point = center + radius * point

    return point


def _make_categorical_sample(X_new, all_neighbors, categories_size, random_state):
    """A support function that populates categorical features' values
    in an artificial point.

    Parameters
    ----------
    X_new : ndarray, shape (n_features, )
        Artificial point to populate categorical features.

    all_neighbors: ndarray, shape (n_features, k_neighbors)
        Nearest neighbors used for majority voting.

    categories_size: list
        Used to tell apart one-hot encoded features.

    random_state : int, RandomState instance or None
        Control the randomization of the algorithm. Used
        for tie breaking when there are two majority values.

    Returns
    -------
    point : ndarray, shape (n_features, )
            Synthetically generated sample.

    """
    for start_idx, end_idx in zip(
        np.cumsum(categories_size)[:-1], np.cumsum(categories_size)[1:]
    ):
        col_maxs = all_neighbors[:, start_idx:end_idx].sum(axis=0)
        # tie breaking argmax
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
    enhanced drop-in replacement for SMOTE as presented in [1]_.

    Read more in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    categorical_features : ndarray of shape (n_cat_features,) or (n_features,)
        Specified which features are categorical. Can either be:

        - array of indices specifying the categorical features;
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.

    {sampling_strategy}

    {random_state}

    truncation_factor : float, optional (default=0.0)
        The type of truncation. The values should be in the [-1.0, 1.0] range.

    deformation_factor : float, optional (default=0.0)
        The type of geometry. The values should be in the [0.0, 1.0] range.

    selection_strategy : str, optional (default='combined')
        The type of Geometric SMOTE algorithm with the following options:
        ``'combined'``, ``'majority'``, ``'minority'``.

    k_neighbors : int or object, optional (default=5)
        If ``int``, number of nearest neighbours to use when synthetic
        samples are constructed for the minority method.  If object, an estimator
        that inherits from :class:`sklearn.neighbors.base.KNeighborsMixin` that
        will be used to find the k_neighbors.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.
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
    random_state_ : instance of RandomState
        If the `random_state` parameter is None, it is a RandomState singleton used by
        np.random. If `random_state` is an int, it is a RandomState instance seeded with
        seed. If `random_state` is already a RandomState instance, it is the same
        object.

    Notes
    -----
    See the original paper: [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [2]_.

    References
    ----------

    .. [1] G. Douzas, F. Bacao, "Geometric SMOTE:
       a geometrically enhanced drop-in replacement for SMOTE",
       Information Sciences, vol. 501, pp. 118-135, 2019.

    .. [2] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique", Journal of Artificial
       Intelligence Research, vol. 16, pp. 321-357, 2002.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from gsmote import GeometricSMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> gsmote = GeometricSMOTE(random_state=1)
    >>> X_res, y_res = gsmote.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})

    """

    def __init__(
        self,
        sampling_strategy='auto',
        random_state=None,
        truncation_factor=1.0,
        deformation_factor=0.0,
        selection_strategy='combined',
        k_neighbors=5,
        categorical_features=None,
        n_jobs=1,
    ):
        super(GeometricSMOTE, self).__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.truncation_factor = truncation_factor
        self.deformation_factor = deformation_factor
        self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the necessary attributes for Geometric SMOTE."""

        # Check random state
        self.random_state_ = check_random_state(self.random_state)

        # Validate strategy
        if self.selection_strategy not in SELECTION_STRATEGY:
            error_msg = (
                'Unknown selection_strategy for Geometric SMOTE algorithm. '
                'Choices are {}. Got {} instead.'
            )
            raise ValueError(
                error_msg.format(SELECTION_STRATEGY, self.selection_strategy)
            )

        # Create nearest neighbors object for positive class
        if self.selection_strategy in ("minority", "combined"):
            self.nns_pos_ = check_neighbors_object(
                'nns_positive', self.k_neighbors, additional_neighbor=1
            )
            self.nns_pos_.set_params(n_jobs=self.n_jobs)

        # Create nearest neighbors object for negative class
        if self.selection_strategy in ("majority", "combined"):
            self.nn_neg_ = check_neighbors_object("nn_negative", nn_object=1)
            self.nn_neg_.set_params(n_jobs=self.n_jobs)

    def _validate_categorical(self):
        """Create the necessary attributes for Geometric SMOTE
        with categorical features"""

        if self.categorical_features is None:
            return self

        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == "bool":
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any(
                [cat not in np.arange(self.n_features_) for cat in categorical_features]
            ):
                raise ValueError(
                    "Some of the categorical indices are out of range. Indices"
                    " should be between 0 and {}".format(self.n_features_)
                )
            self.categorical_features_ = categorical_features
        self.continuous_features_ = np.setdiff1d(
            np.arange(self.n_features_), self.categorical_features_
        )

        if self.categorical_features_.size == self.n_features_in_:
            raise ValueError(
                "GeometricSMOTE is not designed to work only with categorical "
                "features. It requires some numerical features."
            )
        return self

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X, y, reset=True, dtype=None, accept_sparse=["csr", "csc"]
        )

        return X, y, binarize_y

    def _make_geometric_samples(self, X, y, pos_class_label, n_samples):
        """A support function that returns an artificials samples inside
        the geometric region defined by nearest neighbors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : array-like, shape (n_samples, )
            Corresponding label for each sample in X.
        pos_class_label : str or int
            The minority class (positive class) target value.
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        X_new : ndarray, shape (n_samples_new, n_features)
            Synthetically generated samples.
        y_new : ndarray, shape (n_samples_new, )
            Target values for synthetic samples.

        """

        # Return zero new samples
        if n_samples == 0:
            return (
                np.array([], dtype=X.dtype).reshape(0, X.shape[1]),
                np.array([], dtype=y.dtype),
                np.array([], dtype=X.dtype),
            )

        # Select positive class samples
        X_pos = X[y == pos_class_label]

        # Force minority strategy if no negative class samples are present
        self.selection_strategy_ = (
            'minority' if X.shape[0] == X_pos.shape[0] else self.selection_strategy
        )

        # Minority or combined strategy
        if self.selection_strategy_ in ("minority", "combined"):
            self.nns_pos_.fit(X_pos)
            points_pos = self.nns_pos_.kneighbors(X_pos)[1][:, 1:]
            samples_indices = self.random_state_.randint(
                low=0, high=len(points_pos.flatten()), size=n_samples
            )
            rows = np.floor_divide(samples_indices, points_pos.shape[1])
            cols = np.mod(samples_indices, points_pos.shape[1])

        # Majority or combined strategy
        if self.selection_strategy_ in ('majority', 'combined'):
            X_neg = X[y != pos_class_label]
            self.nn_neg_.fit(X_neg)
            points_neg = self.nn_neg_.kneighbors(X_pos)[1]
            if self.selection_strategy_ == "majority":
                samples_indices = self.random_state_.randint(
                    low=0, high=len(points_neg.flatten()), size=n_samples
                )
                rows = np.floor_divide(samples_indices, points_neg.shape[1])
                cols = np.mod(samples_indices, points_neg.shape[1])

        # In the case that the median std was equal to zeros, we have to
        # create non-null entry based on the encoded of OHE
        if self.categorical_features is not None:
            if math.isclose(self.median_std_, 0):
                X[:, self.continuous_features_.size :] = self._X_categorical_encoded
                # Select positive class samples
                X_pos = X[y == pos_class_label]
                if self.selection_strategy_ in ('majority', 'combined'):
                    X_neg = X[y != pos_class_label]

        # Generate new samples
        X_new = np.zeros((n_samples, X.shape[1]))
        all_neighbors_ = []
        for ind, (row, col) in enumerate(zip(rows, cols)):

            # Define center point
            center = X_pos[row]

            # Minority strategy
            if self.selection_strategy_ == "minority":
                surface_point = X_pos[points_pos[row, col]]
                all_neighbors = (
                    (X_pos[points_pos[row]])
                    if self.categorical_features is not None
                    else None
                )

            # Majority strategy
            elif self.selection_strategy_ == "majority":
                surface_point = X_neg[points_neg[row, col]]
                all_neighbors = (
                    (X_neg[points_neg[row]])
                    if self.categorical_features is not None
                    else None
                )

            # Combined strategy
            else:
                surface_point_pos = X_pos[points_pos[row, col]]
                surface_point_neg = X_neg[points_neg[row, 0]]
                radius_pos = norm(center - surface_point_pos)
                radius_neg = norm(center - surface_point_neg)
                surface_point = (
                    surface_point_neg if radius_pos > radius_neg else surface_point_pos
                )
                all_neighbors = (
                    np.vstack([X_pos[points_pos[row]], X_neg[points_neg[row]]])
                    if self.categorical_features is not None
                    else None
                )

            if self.categorical_features is not None:
                all_neighbors_.append(all_neighbors)

            # Append new sample - no categorical features
            X_new[ind] = _make_geometric_sample(
                center,
                surface_point,
                self.truncation_factor,
                self.deformation_factor,
                self.random_state_,
            )

        # Create new samples for target variable
        y_new = np.array([pos_class_label] * len(samples_indices))

        return X_new, y_new, all_neighbors_

    def _make_categorical_samples(self, X_new, y_new, categories_size, all_neighbors_):
        for ind, all_neighbors in enumerate(all_neighbors_):
            # Append new sample - continuous features
            X_new[ind] = _make_categorical_sample(
                X_new[ind], all_neighbors, categories_size, self.random_state_
            )
        return X_new, y_new

    def _encode_categorical(self, X, y):
        """TODO"""
        # compute the median of the standard deviation of the minority class
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        # Separate categorical features from continuous features
        X_continuous = X[:, self.continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=["csr", "csc"])
        X_categorical = X[:, self.categorical_features_].copy()
        X_minority = X_continuous[np.flatnonzero(y == class_minority)]

        if sparse.issparse(X):
            if X.format == "csr":
                _, var = csr_mean_variance_axis0(X_minority)
            else:
                _, var = csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        self.median_std_ = np.median(np.sqrt(var))

        if X_continuous.dtype.name != "object":
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = np.float64
        self.ohe_ = OneHotEncoder(sparse=True, handle_unknown="ignore", dtype=dtype_ohe)

        # the input of the OneHotEncoder needs to be dense
        X_ohe = self.ohe_.fit_transform(
            X_categorical.toarray() if sparse.issparse(X_categorical) else X_categorical
        )

        # we can replace the 1 entries of the categorical features with the
        # median of the standard deviation. It will ensure that whenever
        # distance is computed between 2 samples, the difference will be equal
        # to the median of the standard deviation as in the original paper.

        # In the edge case where the median of the std is equal to 0, the 1s
        # entries will be also nullified. In this case, we store the original
        # categorical encoding which will be later used for inversing the OHE
        if math.isclose(self.median_std_, 0):
            self._X_categorical_encoded = X_ohe.toarray()

        X_ohe.data = np.ones_like(X_ohe.data, dtype=X_ohe.dtype) * self.median_std_ / 2

        if self._issparse:
            X_encoded = np.hstack([X_continuous.toarray(), X_ohe.toarray()])
        else:
            X_encoded = np.hstack([X_continuous, X_ohe.toarray()])

        return X_encoded

    def _decode_categorical(self, X_resampled):
        """Reverses the encoding of the categorical features to match
        the dataset's original structure."""

        if math.isclose(self.median_std_, 0):
            X_resampled[
                : self._X_categorical_encoded.shape[0], self.continuous_features_.size :
            ] = self._X_categorical_encoded

        X_resampled = sparse.csr_matrix(X_resampled)

        X_res_cat = X_resampled[:, self.continuous_features_.size :]
        X_res_cat.data = np.ones_like(X_res_cat.data)
        X_res_cat_dec = self.ohe_.inverse_transform(X_res_cat)

        if self._issparse:
            X_resampled = sparse.hstack(
                (X_resampled[:, : self.continuous_features_.size], X_res_cat_dec),
                format="csr",
            )
        else:
            X_resampled = np.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size].toarray(),
                    X_res_cat_dec,
                )
            )

        indices_reordered = np.argsort(
            np.hstack((self.continuous_features_, self.categorical_features_))
        )

        if sparse.issparse(X_resampled):
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]

        return X_resampled

    def _fit_resample(self, X, y):

        # Save basic data
        self.n_features_ = X.shape[1]
        self._issparse = sparse.issparse(X)
        X_dtype = X.dtype

        # Validate estimator's parameters
        self._validate_categorical()._validate_estimator()

        # Preprocess categorical data
        if self.categorical_features is not None:
            X = self._encode_categorical(X, y)
            categories_size = [self.continuous_features_.size] + [
                cat.size for cat in self.ohe_.categories_
            ]

        # Copy data
        X_resampled, y_resampled = X.copy(), y.copy()

        # Resample
        for class_label, n_samples in self.sampling_strategy_.items():

            # Apply gsmote mechanism
            X_new, y_new, all_neighbors_ = self._make_geometric_samples(
                X, y, class_label, n_samples
            )

            # Apply smotenc mechanism
            if self.categorical_features is not None:
                X_new, y_new = self._make_categorical_samples(
                    X_new, y_new, categories_size, all_neighbors_
                )

            # Append new data
            X_resampled, y_resampled = (
                np.vstack((X_resampled, X_new)),
                np.hstack((y_resampled, y_new)),
            )

        # reverse the encoding of the categorical features
        if self.categorical_features is not None:
            X_resampled = self._decode_categorical(X_resampled).astype(X_dtype)
        else:
            X_resampled = X_resampled.astype(X_dtype)
        return X_resampled, y_resampled
