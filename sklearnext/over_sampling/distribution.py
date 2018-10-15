"""
Distributor classes for clustering oversampling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from collections import Counter, OrderedDict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances


class BaseDistributor(BaseEstimator):

    def __init__(self, labels=None, neighbors=None):
        self.labels = labels
        self.neighbors = neighbors

    def fit(self, X, y):
        """Find the intra-label and inter-label statistics.

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
        self.intra_distribution_, self.inter_distribution_ = [(0, 1.0)], []
        return self

    def apply(self):
        """Distribute the samples intra-label and inter-label."""
        return self.intra_distribution_, self.inter_distribution_


class DensityDistributor(BaseDistributor):

    def __init__(self, labels=None, neighbors=None, filtering_threshold=1.0, distances_exponent=0, sparsity_based=True, distribution_ratio=None):
        super(DensityDistributor, self).__init__(labels=labels, neighbors=neighbors)
        self.filtering_threshold = filtering_threshold
        self.distances_exponent = distances_exponent
        self.sparsity_based = sparsity_based
        self.distribution_ratio = distribution_ratio

    def _calculate_clusters_density(self, X, y):
        """Calculate the density of the filtered clusters."""

        # Generate a combination of cluster and class labels
        majority_label = Counter(y).most_common()[0][0]
        multi_labels = list(zip(self.labels, y == majority_label))

        # Count samples per multilabel
        labels_counts = Counter(multi_labels)
        self.samples_counts_ = OrderedDict()
        for label, count in labels_counts.items():
            self.samples_counts_.setdefault(label[0], []).append((label[1], count))

        # Calculate density
        self.clusters_density_ = dict()
        for cluster_label, ((is_majority, n_samples), *cluster_info) in self.samples_counts_.items():
        
            # Calculate number of majority and minority samples in each cluster
            n_class_samples = (n_samples, cluster_info[0][1]) if cluster_info else (n_samples, 0)
            n_majority_samples, n_minority_samples = n_class_samples if is_majority else n_class_samples[::-1]
        
            # Calculate imbalance ratio
            IR = n_majority_samples / n_minority_samples if n_minority_samples > 0 else np.inf

            # Identify filtered clusters
            if IR < self.filtering_threshold:
                mask = [label == cluster_label and not is_majority for label, is_majority in multi_labels]
                sum_distances = np.triu(euclidean_distances(X[mask])).sum()
                self.clusters_density_[cluster_label] =  n_minority_samples / (sum_distances ** self.distances_exponent) if sum_distances > 0 else np.inf
        
        # Convert infinite densities to finite
        finite_densites = [val for val in self.clusters_density_.values() if not np.isinf(val)]
        max_density = max(finite_densites) if len(finite_densites) > 0 else 1.0
        self.clusters_density_ = {label: float(max_density if np.isinf(density) else density) for label, density in self.clusters_density_.items()}

    def _intra_distribute(self):
        """Distribute the generated samples in each cluster based on their density."""
    
        # Calculate weights
        weights = {label: (1 / density if self.sparsity_based else density) for label, density in self.clusters_density_.items()}
        normalization_factor = sum(weights.values())

        # Distribute generated samples
        self.intra_distribution_ = [(label, self.distribution_ratio_ * weight / normalization_factor) for label, weight in weights.items()]

    def _inter_distribute(self):
        """Distribute the generated samples between clusters based on their density."""
            
        # Sum their density for each pair
        filtered_neighbors = [pair for pair in self.neighbors if pair[0] in self.clusters_density_.keys() and pair[1] in self.clusters_density_.keys()]
        inter_clusters_density = {(label1, label2): (self.clusters_density_[label1] + self.clusters_density_[label2]) for label1, label2 in filtered_neighbors}
    
        # Calculate weights
        weights = {(label1, label2): (1 / density if self.sparsity_based else density) for (label1, label2), density in inter_clusters_density.items()}
        normalization_factor = sum(weights.values())

        # Distribute generated samples
        self.inter_distribution_ = [((label1, label2), (1 - self.distribution_ratio_) * weight / normalization_factor) for (label1, label2), weight in weights.items()]
    
    def fit(self, X, y):
        """Find the intra-label and inter-label statistics based on the minority samples density of the clusters.

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

        super(DensityDistributor, self).fit(X, y)
    
        # Set default distribution ratio
        self.distribution_ratio_ = 1.0 if self.distribution_ratio is None else self.distribution_ratio

        # Check parameters
        error_msgs = {
            'filtering_threshold': 'Parameter `filtering_threshold` should be a non negative number.',
            'distances_exponent': 'Parameter `distances_exponent` should be a non negative number.',
            'sparsity_based': 'Parameter `sparsity_based` should be True or False.',
            'distribution_ratio': 'Parameter `distribution_ratio` should be a number in the range [0.0, 1.0].'
        }

        try:
            if self.filtering_threshold < 0.0:
                raise ValueError(error_msgs['filtering_threshold'])
        except TypeError:
            raise TypeError(error_msgs['filtering_threshold'])

        try:
            if self.distances_exponent < 0.0:
                raise ValueError(error_msgs['distances_exponent'])
        except TypeError:
            raise TypeError(error_msgs['distances_exponent'])

        if not isinstance(self.sparsity_based, bool):
            raise TypeError(error_msgs['sparsity_based'])

        try:
            if self.distribution_ratio_ < 0.0 or self.distribution_ratio_ > 1.0:
                raise ValueError(error_msgs['distribution_ratio'])
            if self.distribution_ratio_ < 1.0 and self.neighbors is None:
                raise ValueError('Parameter `neiborhood` is equal to None, therefore attribute `distribution_ratio` should be equal to 1.0.')
        except TypeError:
            raise TypeError(error_msgs['distribution_ratio'])

        # Fitting process
        if self.labels is not None:

            # Calculate clusters density
            self._calculate_clusters_density(X, y)

            # Intra label distribution
            self._intra_distribute()

            # Inter label distribution
            if self.neighbors is not None:
                self._inter_distribute()
        
        return self