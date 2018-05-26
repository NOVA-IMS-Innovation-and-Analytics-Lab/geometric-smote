"""
The :mod:`sklearnext.model_selection.split` module includes classes and
functions to split the data based on a preset strategy.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
from sklearn.model_selection._split import _BaseKFold, _num_samples
from sklearn.utils import indexable


class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator.

    Parameters
    ----------
    n_splits : int, default=4
        Number of splits. Must be at least 1.
    test_percentage : float, default=0.1
        The percentage of test samples. The values should be
        in the [0.0, 1.0] range.
    """

    def __init__(self, n_splits=4, test_percentage=0.1):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.test_percentage = test_percentage

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
        test_size = int(n_samples * self.test_percentage)
        test_starts = [n_samples - test_size * ind for ind in range(1, self.n_splits + 1)]
        test_ends = [n_samples - test_size * ind for ind in range(0, self.n_splits)]
        for test_start, test_end in zip(test_starts, test_ends):
            train_indices = indices[0:test_start]
            test_indices = indices[test_start:test_end]
            yield (train_indices, test_indices)
