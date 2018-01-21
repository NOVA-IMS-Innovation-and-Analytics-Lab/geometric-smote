"""
This module contains the implementation of the
Conditional Generative Adversarial Network as
an oversampling algorithm.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
from numpy.linalg import norm
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_neighbors_object
from sklearn.utils import check_random_state, safe_indexing
from scipy import sparse
from ..utils import check_random_states

SELECTION_STRATEGY = ('combined', 'majority', 'minority')


def _make_geometric_sample(center, surface_point, truncation_factor=.0, deformation_factor=.0, random_state=None):
    """Returns a generated point based on a center point ,a surface_point
    and three geometric transformations."""

    # Generate a point inside the unit hyper-sphere
    if np.array_equal(center, surface_point):
        return center
    radius = norm(center - surface_point)
    random_state = check_random_state(random_state)
    normal_samples = random_state.normal(size=center.size)
    point_on_unit_sphere = normal_samples / norm(normal_samples)
    point = (random_state.uniform(size=1) ** (1 / center.size)) * point_on_unit_sphere

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

class GeometricSMOTE(BaseOverSampler):
    """Class to perform oversampling using Geometric SMOTE algorithm.

    Parameters
    ----------
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

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
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 truncation_factor=.0,
                 deformation_factor=.0,
                 selection_strategy='combined',
                 k_neighbors=5,
                 n_jobs=1):
        super().__init__(ratio=ratio, random_state=random_state)
        self.truncation_factor = truncation_factor
        self.deformation_factor = deformation_factor
        self.selection_strategy = selection_strategy
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the necessary objects for Geometric SMOTE."""

        if self.selection_strategy not in SELECTION_STRATEGY:
            error_msg = 'Unknown selection_strategy for Geometric SMOTE algorithm. Choices are {}. Got {} instead.'
            raise ValueError(error_msg.format(SELECTION_STRATEGY, self.selection_strategy))

        if self.selection_strategy in ('minority', 'combined'):
            self.nns_pos_ = check_neighbors_object('nns_positive', self.k_neighbors, additional_neighbor=1)
            self.nns_pos_.set_params(n_jobs=self.n_jobs)

        if self.selection_strategy in ('majority', 'combined'):
            self.nn_neg_ = check_neighbors_object('nn_negative', nn_object=1)
            self.nn_neg_.set_params(n_jobs=self.n_jobs)

    def _make_geometric_samples(self, X, y, pos_class_label, n_samples):
        """Generate synthetic samples based on the selection strategy."""
        random_state = check_random_state(self.random_state)
        random_states = check_random_states(self.random_state, n_samples)
        X_pos = safe_indexing(X, np.flatnonzero(y == pos_class_label))
        if self.selection_strategy in ('minority', 'combined'):
            self.nns_pos_.fit(X_pos)
            points_pos = self.nns_pos_.kneighbors(X_pos)[1][:, 1:]
            samples_indices = random_state.randint(low=0, high=len(points_pos.flatten()), size=n_samples)
            rows = np.floor_divide(samples_indices, points_pos.shape[1])
            cols = np.mod(samples_indices, points_pos.shape[1])
        if self.selection_strategy in ('majority', 'combined'):
            X_neg = safe_indexing(X, np.flatnonzero(y != pos_class_label))
            self.nn_neg_.fit(X_neg)
            points_neg = self.nn_neg_.kneighbors(X_pos)[1]
            if self.selection_strategy == 'majority':
                samples_indices = random_state.randint(low=0, high=len(points_neg.flatten()), size=n_samples)
                rows = np.floor_divide(samples_indices, points_neg.shape[1])
                cols = np.mod(samples_indices, points_neg.shape[1])
        X_new = np.zeros((n_samples, X.shape[1]))
        for ind, (row, col, random_state) in enumerate(zip(rows, cols, random_states)):
            if self.selection_strategy == 'minority':
                center = X_pos[row]
                surface_point = X_pos[points_pos[row, col]]
            elif self.selection_strategy == 'majority':
                center = X_pos[row]
                surface_point = X_neg[points_neg[row, col]]
            else:
                center = X_pos[row]
                surface_point_pos = X_pos[points_pos[row, col]]
                surface_point_neg = X_neg[points_neg[row, 0]]
                radius_pos = norm(center - surface_point_pos)
                radius_neg = norm(center - surface_point_neg)
                surface_point = surface_point_neg if radius_pos > radius_neg else surface_point_pos
            X_new[ind] = _make_geometric_sample(center, surface_point, self.truncation_factor, self.deformation_factor, random_state)
        y_new = np.array([pos_class_label] * len(samples_indices))
        return X_new, y_new

    def _sample(self, X, y):
        """Resample the dataset using the Geometric SMOTE algorithm.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
        """

        self._validate_estimator()

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_label, n_samples in self.ratio_.items():

            if n_samples == 0:
                continue

            X_new, y_new = self._make_geometric_samples(X, y, class_label, n_samples)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))

            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled


    