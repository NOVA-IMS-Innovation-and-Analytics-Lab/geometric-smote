"""
This module includes a class to split the time series data
based on a selected strategy.
"""

import numpy as np
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator.

    Parameters
    ----------
    n_splits : int, default=4
        Number of splits. Must be at least 1.
    time_span : int, default=10
        Number of samples of the test set.
    """

    def __init__(self, n_splits=4, time_span=10):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.time_span = time_span

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)
        test_intervals = [(n_samples - (i + 1) * self.time_span, n_samples - i * self.time_span)\
                          for i in range(self.n_splits)]
        for test_start, test_stop in test_intervals:
            train_indices = indices[:test_start]
            test_indices = indices[test_start:test_stop]
            yield (train_indices, test_indices)
