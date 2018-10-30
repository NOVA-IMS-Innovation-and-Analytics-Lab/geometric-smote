"""
The :mod:`sklearnext.preprocessing.data` includes utilities to select
features and sample the input matrix.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
from sklearn.utils import indices_to_mask
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array, check_random_state
from sklearn.feature_selection.univariate_selection import _BaseFilter
from imblearn.base import BaseSampler


def _generate_zero_scores(X, y):
    """Return an array of zero scores."""
    return np.zeros((1, X.shape[1]))


class FeatureSelector(_BaseFilter):
    """Select features according to an array of indices.

    Parameters
    ----------
    indices : array-like, shape = [n_features]
        The selected features.
    """

    def __init__(self, indices=None):
        super(FeatureSelector, self).__init__(_generate_zero_scores)
        self.indices = indices

    def _check_params(self, X, y):
        self.n_features_ = X.shape[1]
        if self.indices is not None:
            self.indices_ = check_array(self.indices, ensure_2d=False)
        else:
            self.indices_ = np.arange(0, self.n_features_)
        if not set(np.arange(self.n_features_)).issuperset(set(self.indices_)):
            raise ValueError("Parameter indices should be an array of any index of the features; Got %r." % self.indices)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        mask = indices_to_mask(self.indices_, self.n_features_)
        return mask


class RowSelector(BaseSampler):
    """Select rows according to a defined percentage.

        Parameters
        ----------
        ratio : float, optional (default=None)
            The ratio of samples to keep. The values should be in the [0.0, 1.0] range.
        selection_strategy : str, int, RandomState instance or None, optional (default=None)
            If str, valid choices are 'head' or 'tail' where the first or last samples
            are used respectively. If int, ``random_state`` is the seed used by
            the random number generator; If ``RandomState`` instance, random_state
            is the random number generator; If ``None``, the random number generator
            is the ``RandomState`` instance used by ``np.random``.
        """
    
    _sampling_type = 'under-sampling'

    def __init__(self, sampling_strategy=None, selection_strategy=None, ratio=None):
        super().__init__(sampling_strategy=sampling_strategy, ratio=ratio)
        self.selection_strategy = selection_strategy

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=None)
        return self

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)
    
    #  define an alias for back-compatibility
    fit_sample = fit_resample

    def _fit_resample(self, X, y):
        """Remove samples from input matrix.

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
            The corresponding rows of `X_resampled`
        """

        self.fit(X, y)

        # Calculate the number of samples to keep
        if self.sampling_strategy is not None and self.sampling_strategy != 1.0:
            self.sampling_strategy_ = self.sampling_strategy
            self.n_samples_ = int(self.sampling_strategy_ * len(X))
        else:
            return X.copy(), y.copy()    
        
        # Return the first rows
        if self.selection_strategy == 'head':
            X_resampled = X[:self.n_samples_].copy()
            y_resampled = y[:self.n_samples_].copy()

        # Return the last rows
        elif self.selection_strategy == 'tail':
            X_resampled = X[-self.n_samples_:].copy()
            y_resampled = y[-self.n_samples_:].copy()

        # Return rows randomly
        else:
            indices = check_random_state(self.selection_strategy).randint(0, len(X), self.n_samples_)
            X_resampled = X[indices].copy()
            y_resampled = y[indices].copy()
            
        return X_resampled, y_resampled
