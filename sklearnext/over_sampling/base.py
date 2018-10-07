"""
Extended base class for oversampling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from abc import abstractmethod
from collections import Counter, OrderedDict

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
    clusters_density = {label: (max_density if np.isinf(density) else density) for label, density in clusters_density.items()}
    
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
    
    # Identify topological neighbors
    topological_neighbors = clusterer.topological_neighbors_
    
    # Sum their density for each pair
    filtered_neighbors = [pair for pair in clusterer.neighbors_ if pair[0] in clusters_density.keys() and pair[1] in clusters_density.keys()]
    inter_clusters_density = {(label1, label2): (clusters_density[label1] + clusters_density[label2]) for label1, label2 in filtered_neighbors}
    
    # Calculate weights
    weights = {(label1, label2): (1 / density if sparsity_based else density) for label, density in inter_clusters_density.items()}
    normalization_factor = sum(weights.values())

    # Distribute generated samples
    distribution = [((label1, label2), (1 - distribution_ratio) * weight / normalization_factor) for (label1, label2), weight in weights.items()]

    return distribution

def generate_distribution(clusterer, X, y, filtering_threshold=1.0, distances_exponent=None, sparsity_based=True, distribution_ratio=None):
    """Distribute the generated samples based on the minority samples density of the clusters."""
    
    # Set default parameters
    distances_exponent = X.shape[1] if distances_exponent is None else distances_exponent
    distribution_ratio = 1.0 if distribution_ratio is None else distribution_ratio

    # Calculate clusters density
    clustering_labels = clusterer.predict(X)
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
        """Validate categotical columns."""
        if self.categorical_cols is not None:
            wrong_input = not isinstance(self.categorical_cols, (list, tuple)) or \
                          len(self.categorical_cols) == 0 or \
                          not set(range(max_col_index)).issuperset(self.categorical_cols)
            if wrong_input:
                error_msg = 'Selected categorical columns should be in the {} range. Got {} instead.'
                raise ValueError(error_msg.format([0, max_col_index - 1], self.categorical_cols))

    def _apply_clustering(self, X, y, **fit_params):
        """Apply clustering on the input space."""
        if self.clusterer is not None:
            self.clusterer.fit(X, y, **fit_params)
    
    def _distribute_samples(self, X, y, **fit_params):
        """Distribute the generated samples on clusters."""
        if self.clusterer is not None:
            self.intra_distribution_, self.inter_distribution_ = self.distribution_function(self.clusterer, X, y, **fit_params)
        else:
            self.intra_distribution_ = [(0, 1.0)]
            self.inter_distribution_ = []

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

        # Validate categorical columns
        self._validate_categorical_cols(X.shape[1])

        # Cluster input space
        self._apply_clustering(X, y, **fit_params)

        # Distribute the generated samples
        self._distribute_samples(X, y, **fit_params)

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

        # Inital ratio
        initial_ratio = self.ratio_.copy()

        # Initialize resampled arrays
        X_resampled = np.array([], dtype=X.dtype).reshape(0, X.shape[1])
        y_resampled = np.array([], dtype=y.dtype)

        # Intracluster oversampling
        self.intra_samples_ = []
        for cluster_label, proportion in self.intra_distribution_:
            self.ratio_ = {class_label:round(n_samples * proportion) for class_label, n_samples in initial_ratio.items()}
            X_resampled_cluster, y_resampled_cluster = self._numerical_sample(X, y)
            X_resampled = np.vstack((X_resampled, X_resampled_cluster))
            y_resampled = np.hstack((y_resampled, y_resampled_cluster))
            self.intra_samples_.append((cluster_label, self.ratio_))

        # Restore initial ratio
        self.ratio_ = initial_ratio

        return X_resampled, y_resampled


