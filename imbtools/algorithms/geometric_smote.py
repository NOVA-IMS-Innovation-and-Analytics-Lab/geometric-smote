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

GEOMETRIC_SMOTE_KIND = ('regular', 'majority')


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
        ``'regular'``, ``'majority'``.

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

        self.nn_k_ = check_neighbors_object('k_neighbors', self.k_neighbors, additional_neighbor=1)
        self.nn_majority_ = check_neighbors_object('nn_majority', 1)
        self.nn_k_.set_params(n_jobs=self.n_jobs)
        self.nn_majority_.set_params(n_jobs=self.n_jobs)

    def _make_geometric_samples(self, X, class_label, nns_data, nns, n_samples):
        """A support function that returns artificial samples constructed in a
        geometric region defined by nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        class_label : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nns_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nns: ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in nn_data.

        n_samples : int
            The number of samples to generate.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray, shape (n_samples_new,)
            Target values for synthetic samples.
        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=len(X), size=n_samples)
        random_states = check_random_states(random_state, len(X))

        radii = nns[0][samples_indices]

        X_new = np.zeros((n_samples, X.shape[1]))
        for row_ind, (center, radius, random_state) in enumerate(zip(X[samples_indices], radii, random_states)):
            X_new[row_ind] = _make_geometric_sample(center, radius, random_state)
        y_new = np.array([class_label] * len(samples_indices))

        return X_new, y_new

    def _sample_regular(self, X, y):
        pass

    def _sample_majority(self, X, y):
        """Resample the dataset using the Geometric SMOTE algorithm and only the
        majority class instances to calculate the geometric radius.

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

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_label, n_samples in self.ratio_.items():

            if n_samples == 0:
                continue

            X_positive = safe_indexing(X, np.flatnonzero(y == class_label))
            X_negative = safe_indexing(X, np.flatnonzero(y != class_label))

            self.nn_majority_.fit(X_negative)
            nns_majority = self.nn_majority_.kneighbors(X_positive)

            X_new, y_new = self._make_geometric_samples(X_positive, class_label, X_negative, nns_majority, n_samples)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled

    def _sample(self, X, y):
        """Resample the dataset.
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

        if self.kind == 'regular':
            return self._sample_regular(X, y)
        elif self.kind == 'majority':
            return self._sample_majority(X, y)

    