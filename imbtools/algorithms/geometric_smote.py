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

GEOMETRIC_SMOTE_KIND = ('regular', 'majority', 'minority')


def _make_geometric_sample(center, radius, random_state=None):
    random_state = check_random_state(random_state)
    normal_samples = random_state.normal(size=center.size)
    on_sphere = normal_samples / norm(normal_samples)
    in_ball = (random_state.uniform(size=1) ** (1 / center.size)) * on_sphere
    return center + radius * in_ball

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

    geometry_factor : float, optional (default=1.0)
        The type of geometry. The values should be in the [0.0, 1.0] range.

    kind : str, optional (default='regular')
        The type of Geometric SMOTE algorithm with the following options:
        ``'regular'``, ``'majority'``, ``'minority'``.

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
                 geometry_factor=1.0,
                 kind='regular',
                 k_neighbors=5,
                 n_jobs=1):
        super().__init__(ratio=ratio, random_state=random_state)
        self.geometry_factor = geometry_factor
        self.kind = kind
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the necessary objects for Geometric SMOTE."""

        if self.kind not in GEOMETRIC_SMOTE_KIND:
            error_msg = 'Unknown kind for Geometric SMOTE algorithm. Choices are {}. Got {} instead.'
            raise ValueError(error_msg.format(GEOMETRIC_SMOTE_KIND, self.kind))

        if self.kind in ('minority', 'regular'):
            self.nns_pos_ = check_neighbors_object('nns_positive', self.k_neighbors, additional_neighbor=1)
            self.nns_pos_.set_params(n_jobs=self.n_jobs)

        if self.kind in ('majority', 'regular'):
            self.nn_neg_ = check_neighbors_object('nn_negative', nn_object=1)
            self.nn_neg_.set_params(n_jobs=self.n_jobs)

    def _make_geometric_samples(self, X, y, pos_class_label, n_samples):
        random_state = check_random_state(self.random_state)
        random_states = check_random_states(self.random_state, n_samples)
        X_pos = safe_indexing(X, np.flatnonzero(y == pos_class_label))
        if self.kind in ('minority', 'regular'):
            self.nns_pos_.fit(X_pos)
            radii_pos = self.nns_pos_.kneighbors(X_pos)[0][:, 1:]
            samples_indices = random_state.randint(low=0, high=len(radii_pos.flatten()), size=n_samples)
            rows = np.floor_divide(samples_indices, radii_pos.shape[1])
            cols = np.mod(samples_indices, radii_pos.shape[1])
        if self.kind in ('majority', 'regular'):
            X_neg = safe_indexing(X, np.flatnonzero(y != pos_class_label))
            self.nn_neg_.fit(X_neg)
            radii_neg = self.nn_neg_.kneighbors(X_pos)[0]
            if self.kind == 'majority':
                samples_indices = random_state.randint(low=0, high=len(radii_neg.flatten()), size=n_samples)
                rows = np.floor_divide(samples_indices, radii_neg.shape[1])
                cols = np.mod(samples_indices, radii_neg.shape[1])
        X_new = np.zeros((n_samples, X.shape[1]))
        for ind, (row, col, random_state) in enumerate(zip(rows, cols, random_states)):
            if self.kind == 'minority':
                radius = radii_pos[row, col]
            elif self.kind == 'majority':
                radius = radii_neg[row, col]
            else:
                radius = min(radii_neg[row, 0], radii_pos[row, col])
            X_new[ind] = _make_geometric_sample(X_pos[row], radius, random_state)
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


    