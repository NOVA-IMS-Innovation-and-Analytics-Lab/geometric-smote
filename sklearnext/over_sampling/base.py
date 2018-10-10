"""
Extended base class for oversampling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from abc import abstractmethod
from collections import Counter, OrderedDict
from inspect import signature
from math import ceil

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.metrics.pairwise import euclidean_distances
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_ratio, check_target_type, hash_X_y


def _count_clusters_samples(labels):
    """Count the minority and majority samples in each cluster."""
    labels_counts = Counter(labels)
    samples_counts = OrderedDict()
    for label, count in labels_counts.items():
        samples_counts.setdefault(label[0], []).append((label[1], count))
    return samples_counts


def _calculate_clusters_density(clustering_labels, X, y, filtering_threshold, distances_exponent):
    """Calculate the density of the filtered clusters."""
    
    # Calculate clusters distribution
    majority_label = Counter(y).most_common()[0][0]
    labels = list(zip(clustering_labels, y == majority_label))
    samples_counts = _count_clusters_samples(labels)

    # Calculate density
    clusters_density = dict()
    for cluster_label, ((is_majority, n_samples), *cluster_info) in samples_counts.items():
        
        # Calculate number of majority and minority samples in each cluster
        n_class_samples = (n_samples, cluster_info[0][1]) if cluster_info else (n_samples, 0)
        n_majority_samples, n_minority_samples = n_class_samples if is_majority else n_class_samples[::-1]
        
        # Calculate imbalance ratio
        IR = n_majority_samples / n_minority_samples if n_minority_samples > 0 else np.inf

        # Identify filtered clusters
        if IR < filtering_threshold:
            mask = [label == cluster_label and not is_majority for label, is_majority in labels]
            sum_distances = np.triu(euclidean_distances(X[mask])).sum()
            clusters_density[cluster_label] =  n_minority_samples / (sum_distances ** distances_exponent) if sum_distances > 0 else np.inf
        
    # Convert infinite densities to finite
    finite_densites = [val for val in clusters_density.values() if not np.isinf(val)]
    max_density = max(finite_densites) if len(finite_densites) > 0 else 1.0
    clusters_density = {label: float(max_density if np.isinf(density) else density) for label, density in clusters_density.items()}
    
    return clusters_density


def _intra_distribute(clusters_density, sparsity_based, distribution_ratio):
    """Distribute the generated samples in each cluster based on their density."""
    
    # Calculate weights
    weights = {label: (1 / density if sparsity_based else density) for label, density in clusters_density.items()}
    normalization_factor = sum(weights.values())

    # Distribute generated samples
    distribution = [(label, distribution_ratio * weight / normalization_factor) for label, weight in weights.items()]
    
    return distribution

def _inter_distribute(clusterer, clusters_density, sparsity_based, distribution_ratio):
    """Distribute the generated samples between clusters based on their density."""
    
    # Sum their density for each pair
    filtered_neighbors = [pair for pair in clusterer.neighbors_ if pair[0] in clusters_density.keys() and pair[1] in clusters_density.keys()]
    inter_clusters_density = {(label1, label2): (clusters_density[label1] + clusters_density[label2]) for label1, label2 in filtered_neighbors}
    
    # Calculate weights
    weights = {(label1, label2): (1 / density if sparsity_based else density) for (label1, label2), density in inter_clusters_density.items()}
    normalization_factor = sum(weights.values())

    # Distribute generated samples
    distribution = [((label1, label2), (1 - distribution_ratio) * weight / normalization_factor) for (label1, label2), weight in weights.items()]

    return distribution

def density_distribution(clusterer, X, y, filtering_threshold=1.0, distances_exponent=None, sparsity_based=True, distribution_ratio=None):
    """Distribute the generated samples based on the minority samples density of the clusters."""
    
    # Set default parameters
    distances_exponent = X.shape[1] if distances_exponent is None else distances_exponent
    distribution_ratio = 1.0 if distribution_ratio is None else distribution_ratio

    # Check parameters
    error_msgs = {
        'filtering_threshold': 'Parameter `filtering_threshold` should be a non negative number.',
        'distances_exponent': 'Parameter `distances_exponent` should be a non negative number.',
        'sparsity_based': 'Parameter `sparsity_based` should be True or False.',
        'distribution_ratio': 'Parameter `distribution_ratio` should be a number in the range [0.0, 1.0].'
    }

    try:
        if filtering_threshold < 0.0:
            raise ValueError(error_msgs['filtering_threshold'])
    except TypeError:
        raise TypeError(error_msgs['filtering_threshold'])

    try:
        if distances_exponent < 0.0:
            raise ValueError(error_msgs['distances_exponent'])
    except TypeError:
        raise TypeError(error_msgs['distances_exponent'])

    if not isinstance(sparsity_based, bool):
        raise TypeError(error_msgs['sparsity_based'])

    try:
        if distribution_ratio < 0.0 or distribution_ratio > 1.0:
            raise ValueError(error_msgs['distribution_ratio'])
        if distribution_ratio < 1.0 and not hasattr(clusterer, 'neighbors_'):
            raise ValueError('Clusterer does not define a neighborhood structure, i.e. attribute `neiborhood_` not found. Parameter `distribution_ratio` should be set to 1.0.')
    except TypeError:
        raise TypeError(error_msgs['distribution_ratio'])

    # Calculate clusters density
    clustering_labels = clusterer.labels_
    clusters_density = _calculate_clusters_density(clustering_labels, X, y, filtering_threshold, distances_exponent)

    # Calculate intracluster distribution
    intra_distribution = _intra_distribute(clusters_density, sparsity_based, distribution_ratio)

    # Calculate intercluster distribution
    inter_distribution = _inter_distribute(clusterer, clusters_density, sparsity_based, distribution_ratio)

    return intra_distribution, inter_distribution


class ExtendedBaseOverSampler(BaseOverSampler, _BaseComposition):
    """An extension of the base class for over-sampling algorithms to
    handle integer and categorical features as well as a .

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 sampling_type=None,
                 categorical_cols=None,
                 clusterer=None,
                 distribution_function=None):
        super(ExtendedBaseOverSampler, self).__init__(ratio, random_state, sampling_type)
        self.categorical_cols = categorical_cols
        self.clusterer = clusterer
        self.distribution_function = distribution_function

    def _validate_categorical_cols(self, max_col_index):
        """Validate categorical columns."""
        if self.categorical_cols is not None:
            wrong_type = not isinstance(self.categorical_cols, (list, tuple)) or len(self.categorical_cols) == 0              
            wrong_range = not set(range(max_col_index)).issuperset(self.categorical_cols)
            error_msg = 'Selected categorical columns should be in the {} range. Got {} instead.'
            if wrong_type:    
                raise TypeError(error_msg.format([0, max_col_index - 1], self.categorical_cols))
            elif wrong_range:
                raise ValueError(error_msg.format([0, max_col_index - 1], self.categorical_cols))

    def _apply_clustering(self, X, y, **fit_params):
        """Apply clustering on the input space."""
        if self.clusterer is not None:
            self.clusterer.fit(X, y, **fit_params)
    
    def _distribute_samples(self, X, y, **fit_params):
        """Distribute the generated samples on clusters."""
        if self.clusterer is not None and hasattr(self.clusterer, 'neighbors_'):
            self.intra_distribution_, self.inter_distribution_ = self.distribution_function_(self.clusterer, X, y, **fit_params)
        else:
            self.intra_distribution_, self.inter_distribution_ = [(0, 1.0)], []
            

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
        super(ExtendedBaseOverSampler, self)._set_params('clusterer', **params)
        return self

    def get_params(self, deep=True):
        """Get the parameters.
        Parameters
        ----------
        deep: bool
            Setting it to True gets the various estimators and the parameters
            of the estimators as well
        """
        return super(ExtendedBaseOverSampler, self)._get_params('clusterer', deep=deep)

    def fit(self, X, y, **fit_params):
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
        super(ExtendedBaseOverSampler, self).fit(X, y)

        # Check distribution function
        self.distribution_function_ = self.distribution_function if self.distribution_function is not None else density_distribution

        # Validate categorical columns
        self._validate_categorical_cols(X.shape[1])

        # Split fit params
        var_names = signature(self.distribution_function_).parameters.keys()
        fit_params_distribute = {param: value for param, value in fit_params.items() if param in var_names}
        fit_params_clustering = {param: value for param, value in fit_params.items() if param not in var_names}

        # Cluster input space
        self._apply_clustering(X, y, **fit_params_clustering)

        # Distribute the generated samples
        self._distribute_samples(X, y, **fit_params_distribute)

        return self

    @abstractmethod
    def _numerical_sample(self, X, y):
        """Resample the numerical features of the dataset
        using the Geometric SMOTE algorithm.

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

        # No clustering is applied
        if self.clusterer is None:
            return self._numerical_sample(X, y)

        # Inital ratio
        initial_ratio = self.ratio_.copy()

        # Initialize resampled arrays
        X_resampled = np.array([], dtype=X.dtype).reshape(0, X.shape[1])
        y_resampled = np.array([], dtype=y.dtype)
        
        # Intracluster oversampling
        self.intra_ratios_ = []
        for label, proportion in self.intra_distribution_:
            
            # Deal with cases where the number of neighbors is greater than the number of samples in cluster
            mask = (self.clusterer.labels_ == label)
            oversampler_uses_knn = hasattr(self, 'k_neighbors')
            n_samples, min_n_samples = mask.sum(), Counter(y[mask]).most_common()[-1][1]
            factor = ceil(self.k_neighbors / min_n_samples) + 1 if oversampler_uses_knn and self.k_neighbors >= min_n_samples else 1
            
            # Filter in cluster data
            X_in_cluster, y_in_cluster = np.vstack([X[mask]] * factor), np.hstack([y[mask]] * factor)

            # Modify ratio
            target_n_samples = sum(initial_ratio.values())
            feasible_n_samples = sum({class_label: n_samples for class_label, n_samples in initial_ratio.items() if class_label in y_in_cluster}.values())
            proportion = (target_n_samples / feasible_n_samples) * proportion
            self.ratio_ = {class_label: round(n_samples * proportion) for class_label, n_samples in initial_ratio.items() if class_label in y_in_cluster}

            # Resample data
            X_resampled_cluster, y_resampled_cluster = self._numerical_sample(X_in_cluster, y_in_cluster)
            X_resampled = np.vstack((X_resampled, X_resampled_cluster[(n_samples * (factor - 1)):]))
            y_resampled = np.hstack((y_resampled, y_resampled_cluster[(n_samples * (factor - 1)):]))
            
            self.intra_ratios_.append((label, self.ratio_.copy()))
        
        # Add non cluster data
        labels, _ = zip(*self.intra_distribution_)
        mask = ~np.isin(self.clusterer.labels_, labels)
        X_resampled, y_resampled = np.vstack((X_resampled, X[mask])), np.hstack((y_resampled, y[mask]))

        # Restore initial ratio
        self.ratio_ = initial_ratio

        return X_resampled, y_resampled

