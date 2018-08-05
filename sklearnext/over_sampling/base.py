"""
Extended base class for oversampling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from abc import abstractmethod
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.metaestimators import _BaseComposition
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_ratio, check_target_type, hash_X_y


def _generate_classes_stats(y, majority_label, imbalance_ratio_threshold, k_neighbors):
    """Generate stats for the various minority classes."""
    counter = Counter(y)
    stats = {label:((counter[majority_label]/ n_samples), n_samples) for label, n_samples in counter.items()
             if label != majority_label}
    include_group = any([ir < imbalance_ratio_threshold and n_samples > k_neighbors
                         if k_neighbors is not None else ir < imbalance_ratio_threshold
                         for label, (ir, n_samples) in stats.items()])
    modified_imbalance_ratios = {label: ((counter[majority_label] + 1) / (n_samples + 1))
                                 for label, n_samples in counter.items()
                                 if label != majority_label}
    return include_group, modified_imbalance_ratios


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
                 integer_cols=None,
                 categorical_cols=None,
                 categorical_ir_threshold=1.0):
        super(ExtendedBaseOverSampler, self).__init__(ratio, random_state, sampling_type)
        self.integer_cols = integer_cols
        self.categorical_cols = categorical_cols
        self.categorical_ir_threshold = categorical_ir_threshold

    @abstractmethod
    def _numerical_sample(self, X, y):
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

    def _integer_sample(self, X, y):
        """Resample the integer features of the dataset.

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

        X_resampled, y_resampled = self._numerical_sample(X, y)
        if self.integer_cols is not None:
            try:
                if len(self.integer_cols) == 0 or not set(range(self.max_col_index_)).issuperset(self.integer_cols):
                    error_msg = 'Selected integer columns should be in the {} range. Got {} instead.'
                    raise ValueError(error_msg.format([0, self.max_col_index_], self.integer_cols))
            except:
                raise ValueError('Parameter `integer_cols` should be a list or tuple in the %s range.' % [0, self.max_col_index_])
            X_resampled[:, self.integer_cols] = np.round(X_resampled[:, self.integer_cols]).astype(int)
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

        self.random_state_ = check_random_state(self.random_state)
        self.max_col_index_ = X.shape[1]

        if self.categorical_cols is None:
            return self._integer_sample(X, y)

        try:
            if len(self.categorical_cols) == 0 or not set(range(self.max_col_index_)).issuperset(self.categorical_cols):
                error_msg = 'Selected categorical columns should be in the {} range. Got {} instead.'
                raise ValueError(error_msg.format([0, self.max_col_index_], self.categorical_cols))
        except:
            raise ValueError('Parameter `categorical_cols` should be a list or tuple in the %s range.' % [0, self.max_col_index_])

        if self.integer_cols is not None and not set(self.integer_cols).isdisjoint(self.categorical_cols):
            raise ValueError('Parameters `integer_cols` and `categorical_cols` should not have common elements.')

        if self.categorical_ir_threshold <= 0.0:
            raise ValueError('Parameter `categorical_threshold` should be a positive number.')

        df = pd.DataFrame(np.column_stack((X, y)))

        # Select groups to oversample
        majority_label = [label for label, n_samples in self.ratio_.items() if n_samples == 0][0]
        minority_labels = [label for label in self.ratio_.keys() if label != majority_label]
        classes_stats = df.groupby(self.categorical_cols, as_index=False).agg(
            {
                df.columns[-1]: lambda y: _generate_classes_stats(
                    y,
                    majority_label,
                    self.categorical_ir_threshold,
                    self.k_neighbors if hasattr(self, 'k_neighbors') else None
                )
            }
        )
        boolean_mask = classes_stats.iloc[:, -1].apply(lambda stat: stat[0])
        included_groups = classes_stats[boolean_mask].iloc[:, :-1].reset_index(drop=True)
        self.n_oversampled_groups_ = len(included_groups)

        # Calculate oversampling weights
        imbalance_ratios = classes_stats[boolean_mask].iloc[:, -1].apply(lambda stat: stat[1]).reset_index(drop=True)
        weights = pd.DataFrame()
        for label in minority_labels:
            label_weights = imbalance_ratios.apply(lambda ratio: ratio.get(label, np.nan))
            label_weights = label_weights / label_weights.sum()
            label_weights.rename(label, inplace=True)
            weights = pd.concat([weights, label_weights], axis=1)

        initial_ratio = self.ratio_.copy()

        X_resampled = pd.DataFrame(columns=df.columns[:-1])
        y_resampled = pd.DataFrame(columns=[df.columns[-1]])

        # Oversample data in each group
        for group_values, (_, weight) in zip(included_groups.values.tolist(), weights.iterrows()):

            # Define ratio in group
            self.ratio_ = {label: (int(n_samples * weight[label]) if label != majority_label else n_samples)
                           for label, n_samples in initial_ratio.items()}

            # Select data in group
            df_group = pd.merge(df, pd.DataFrame([group_values], columns=self.categorical_cols)).drop(columns=self.categorical_cols)
            X_group, y_group = df_group.iloc[:, :-1], df_group.iloc[:, -1]

            # Oversample data
            X_group_resampled, y_group_resampled = self._integer_sample(X_group.values, y_group.values)
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
        excluded_groups = classes_stats[~boolean_mask].iloc[:, :-1].reset_index(drop=True)
        df_excluded = pd.merge(df, excluded_groups)
        X_resampled = X_resampled.append(df_excluded.iloc[:, :-1]).values
        y_resampled = y_resampled.append(df_excluded.iloc[:, -1:]).values.reshape(-1)

        return X_resampled, y_resampled


class ClusteringBaseOverSampler(ExtendedBaseOverSampler, _BaseComposition):

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 sampling_type=None,
                 integer_cols=None,
                 categorical_cols=None,
                 categorical_ir_threshold=1.0,
                 clusterer=None,
                 clustering_ir_threshold=1.0,
                 density_exponent=None,
                 cluster_topology=None):
        super(ClusteringBaseOverSampler, self).__init__(ratio, random_state, sampling_type, integer_cols,
                                                        categorical_cols, categorical_ir_threshold)
        self.clusterer = clusterer
        self.clustering_ir_threshold = clustering_ir_threshold
        self.density_exponent = density_exponent
        self.cluster_topology = cluster_topology

    def set_params(self, **params):
        """Set the parameters.
        Valid parameter keys can be listed with get_params().
        Parameters
        ----------
        params : keyword arguments
            Specific parameters using e.g. set_params(parameter_name=new_value)
            In addition, to setting the parameters of the ``_ParametrizedEstimatorsMixin``,
            the individual estimators of the ``_ParametrizedEstimatorsMixin`` can also be
            set or replaced by setting them to None.
        """
        super(ClusteringBaseOverSampler, self)._set_params('clusterer', **params)
        return self

    def get_params(self, deep=True):
        """Get the parameters.
        Parameters
        ----------
        deep: bool
            Setting it to True gets the various estimators and the parameters
            of the estimators as well
        """
        return super(ClusteringBaseOverSampler, self)._get_params('clusterer', deep=deep)

    def _numerical_sample(self, X, y):
        pass

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        y = check_target_type(y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        self.ratio_ = check_ratio(self.ratio, y, self._sampling_type)

        # Cluster input space
        self.clustering_labels_ = self.clusterer[0][1].fit_predict(X, y)

        # Identify majority and minority
        majority_label = [label for label, n_samples in self.ratio_.items() if n_samples == 0][0]
        minority_labels = [label for label in self.ratio_.keys() if label != majority_label]

        # Clusters imbalance ratios

        weights = pd.DataFrame()


        return self



