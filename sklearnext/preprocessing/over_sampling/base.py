"""
Extended base class for oversampling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from math import ceil
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from imblearn.over_sampling.base import BaseOverSampler

CATEGORICAL_STRATEGY = ('most_frequent', 'less_frequent', 'random')


class ExtendedBaseOverSampler(BaseOverSampler):
    """An extension of the base class for over-sampling algorithms to
    handle categorical features.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 sampling_type=None,
                 categorical_cols=None,
                 categorical_threshold=1.0,
                 categorical_strategy='most_frequent'):
        super(ExtendedBaseOverSampler, self).__init__(ratio, random_state, sampling_type)
        self.categorical_cols = categorical_cols
        self.categorical_threshold = categorical_threshold
        self.categorical_strategy = categorical_strategy

    @abstractmethod
    def _group_sample(self, X, y):
        """Resample the numerical features of the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the numerical data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            The array containing the numerical resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`
        """
        pass

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

        self.random_state_ = check_random_state(self.random_state)

        if self.categorical_cols is None:
            return self._group_sample(X, y)
        max_col_index = X.shape[1]
        try:
            if len(self.categorical_cols) == 0 or not set(range(max_col_index)).issuperset(self.categorical_cols):
                error_msg = 'Selected categorical columns should be in the {} range. Got {} instead.'
                raise ValueError(error_msg.format([0, max_col_index], self.categorical_cols))
        except:
            raise ValueError('Parameter `categorical_cols` should be a list or tuple in the %s range.' % [0, max_col_index])

        if self.categorical_strategy not in CATEGORICAL_STRATEGY:
            error_msg = 'Unknown categorical strategy. Choices are {}. Got {} instead.'
            raise ValueError(error_msg.format(CATEGORICAL_STRATEGY, self.categorical_strategy))

        if self.categorical_threshold <= 0.0 or self.categorical_threshold > 1.0:
            raise ValueError('Parameter `categorical_threshold` should be in the (0, 1] range.')

        df = pd.DataFrame(np.column_stack((X, y)))

        # Count samples in each group
        n_samples_groups = df.groupby(self.categorical_cols).size().sort_values(ascending=False)

        # Select number of groups based on threshold
        n_groups = ceil(len(n_samples_groups) * self.categorical_threshold)

        # Define data to exclude from oversampling
        excluded_groups = pd.DataFrame(n_samples_groups[n_groups:]).reset_index().iloc[:, :-1]
        df_excluded = pd.merge(df, excluded_groups, how='right')
        X_excluded, y_excluded = df_excluded.iloc[:, :-1], df_excluded.iloc[:, -1:]

        # Selected groups to oversample data based on different strategies
        n_samples_groups = n_samples_groups[:n_groups]
        weights = n_samples_groups / n_samples_groups.sum()
        if self.categorical_strategy == 'less_frequent':
            weights[:] = weights.values[::-1]
        elif self.categorical_strategy == 'random':
            weights = weights.sample(weights.size, random_state=self.random_state_)

        initial_ratio = self.ratio_.copy()

        X_resampled = pd.DataFrame(columns=df.columns[:-1])
        y_resampled = pd.DataFrame(columns=[df.columns[-1]])

        # Oversample data in each group
        for group_values, weight in weights.items():

            # Define ratio in group
            self.ratio_ = {class_label: round(n_samples * weight) for class_label, n_samples in initial_ratio.items()}

            # Select data in group
            df_group = pd.merge(df, pd.DataFrame([group_values], columns=self.categorical_cols)).drop(columns=self.categorical_cols)
            X_group, y_group = df_group.iloc[:, :-1], df_group.iloc[:, -1]

            # Oversample data
            X_group_resampled, y_group_resampled = self._group_sample(X_group.values, y_group.values)
            X_group_categorical = np.array(group_values * len(X_group_resampled)).reshape(len(X_group_resampled), -1)
            X_group_resampled = np.column_stack((X_group_resampled, X_group_categorical))
            X_group_resampled = pd.DataFrame(X_group_resampled, columns=list(X_group.columns) + self.categorical_cols)
            y_group_resampled = pd.DataFrame(y_group_resampled, columns=y_resampled.columns)

            # Append resampled data
            X_resampled = X_resampled.append(X_group_resampled.loc[:, X_resampled.columns])
            y_resampled = y_resampled.append(y_group_resampled)

        # Restore ratio
        self.ratio_ = initial_ratio.copy()

        # Append excluded data
        X_resampled = X_resampled.append(X_excluded).values
        y_resampled = y_resampled.append(y_excluded).values.reshape(-1)

        return X_resampled, y_resampled

