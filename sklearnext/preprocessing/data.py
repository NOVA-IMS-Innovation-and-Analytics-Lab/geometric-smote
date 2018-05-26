"""
The :mod:`sklearnext.preprocessing.data` includes utilities to select
features and sample the input matrix.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import warnings
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils import indices_to_mask
from sklearn.utils.validation import check_is_fitted, check_array, check_random_state
from sklearn.feature_selection.univariate_selection import _BaseFilter
from sklearn.externals.six import string_types


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
        self.num_features_ = X.shape[1]
        if self.indices is not None:
            self.indices_ = check_array(self.indices, ensure_2d=False)
        else:
            self.indices_ = self.num_features_ // 2
        if not set(np.arange(self.num_features_)).issuperset(set(self.indices_)):
            raise ValueError("Parameter indices should be an array of any index of the features; Got %r." % self.indices)

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        mask = indices_to_mask(self.indices_, self.num_features_)
        return mask


class RowSelector(TransformerMixin):
    """Select rows according to a defined percentage.

        Parameters
        ----------
        percentage : float, optional (default=1.0)
            The percentage of samples to keep. The values should
            be in the [0.0, 1.0] range.
        random_state : str, int, RandomState instance or None, optional (default=None)
            If str, valid choices are 'head' or 'tail' where the first or last samples
            are used respectively. If int, ``random_state`` is the seed used by
            the random number generator; If ``RandomState`` instance, random_state
            is the random number generator; If ``None``, the random number generator
            is the ``RandomState`` instance used by ``np.random``.
        """

    def __init__(self, percentage=1.0, random_state=None):
        self.percentage = percentage
        self.random_state = random_state

    def fit(self, X, y=None):
        """Save the initial input matrix and the number of samples
        to be removed.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The input matrix.

        y : Passthrough for ``Pipeline`` compatibility.
        """
        self.X_ = check_array(X)
        self.n_samples_ = int(self.percentage * len(X)) if self.percentage < 1.0 else None
        return self

    def transform(self, X, y='deprecated', copy=None):
        """Remove samples from input matrix.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The input matrix.
        y : (ignored)
            .. deprecated:: 0.19
               This parameter will be removed in 0.21.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        check_is_fitted(self, 'n_samples_')
        X = check_array(X)
        if np.array_equal(X, self.X_) and self.n_samples_ is not None:
            if self.random_state == 'head':
                self.X_t_ = X[:self.n_samples_].copy()
            elif self.random_state == 'tail':
                self.X_t_ = X[-self.n_samples_:].copy()
            else:
                random_state = check_random_state(self.random_state)
                self.X_t_ = X[random_state.randint(0, len(X), self.n_samples_)].copy()
            return self.X_t_
        return X

    def inverse_transform(self, X):
        """Return the initial input matrix.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The input matrix.
        y : (ignored)
            .. deprecated:: 0.19
               This parameter will be removed in 0.21.
        """
        if hasattr(self, "X_t_") and np.array_equal(self.X_t_, X):
            return self.X_
        else:
            return X




