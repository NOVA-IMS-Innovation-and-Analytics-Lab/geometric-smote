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
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state, check_X_y
from sklearn.metrics.pairwise import euclidean_distances
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_ratio, check_target_type, hash_X_y, check_neighbors_object


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

        if self.labels is None:
            return self
    
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
        self._calculate_clusters_density(X, y)
        self._intra_distribute()
        if self.neighbors is not None:
            self._inter_distribute()
        
        return self


class ExtendedBaseOverSampler(BaseOverSampler):
    """An extension of the base class for over-sampling algorithms to
    handle integer and categorical features as well as clustering based 
    oversampling.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 sampling_type=None,
                 categorical_cols=None,
                 clusterer=None,
                 distributor=None):
        super(ExtendedBaseOverSampler, self).__init__(ratio, random_state, sampling_type)
        self.categorical_cols = categorical_cols
        self.clusterer = clusterer
        self.distributor = distributor

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

    def _apply_clustering_distribution(self, X, y, **fit_params):
        """Apply clustering on the input space and distribute generated samples."""
        
        # Check distributor
        self.distributor_ = DensityDistributor() if self.distributor is None else clone(self.distributor)
        
        # Fit clusterer
        if self.clusterer is not None:
            
            # Check clusterer
            self.clusterer_ = clone(self.clusterer).fit(X, y, **fit_params)
        
            # Set labels and neighbors
            if hasattr(self.clusterer_, 'neighbors_'):
                self.distributor_.set_params(labels=self.clusterer_.labels_, neighbors=self.clusterer_.neighbors_)
            else:
                self.distributor_.set_params(labels=self.clusterer_.labels_)

        # Fit distributor
        self.distributor_.fit(X, y)

    def _modify_attributes(self, n_minority_samples, n_samples):
        """Modify attributes for corner cases."""
        
        initial_attributes = {}
        
        # Use random oversampling
        if n_minority_samples == 1:
            initial_attributes['_basic_sample'] = self._basic_sample
            random_oversampler = RandomOverSampler()
            random_oversampler.ratio_ = self.ratio_.copy()
            self._basic_sample = random_oversampler._sample
        
        # Decrease number of nearest neighbors
        elif hasattr(self, 'k_neighbors') and n_minority_samples <= self.k_neighbors:
            initial_attributes['k_neighbors'] = self.k_neighbors
            self.k_neighbors = n_minority_samples - 1
        elif hasattr(self, 'n_neighbors') and n_minority_samples <= self.n_neighbors:
            initial_attributes['n_neighbors'] = self.n_neighbors
            self.n_neighbors = n_minority_samples - 1
        if n_minority_samples > 1 and hasattr(self, 'm_neighbors') and n_samples <= self.m_neighbors:
            initial_attributes['m_neighbors'] = self.m_neighbors
            self.m_neighbors = n_samples - 1

        return initial_attributes

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

        # Cluster input space and distribute samples
        self._apply_clustering_distribution(X, y, **fit_params)

        return self

    @abstractmethod
    def _basic_sample(self, X, y):
        """Basic resample of the dataset.

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

    def _intra_sample(self, X, y, initial_ratio):
        """Intracluster resampling."""

        # Initialize arrays of new data
        X_new = np.array([], dtype=X.dtype).reshape(0, X.shape[1])
        y_new = np.array([], dtype=y.dtype)
        
        # Intracluster oversampling
        self.intra_ratios_ = []
        for label, proportion in self.distributor_.intra_distribution_:
            
            # Filter data in cluster
            if self.clusterer is not None:
                mask = (self.clusterer_.labels_ == label)
                X_in_cluster, y_in_cluster = X[mask], y[mask]
            else:
                X_in_cluster, y_in_cluster = X, y

            # Calculate ratio in the cluster
            cluster_ratio = {class_label: (round(n_samples * proportion) if class_label in y_in_cluster else 0)
                             for class_label, n_samples in initial_ratio.items()}

            # Count in cluster target variable
            y_cluster_count = Counter(y_in_cluster)

            # Resample data
            for class_label, n_samples in cluster_ratio.items():

                # Modify ratio
                self.ratio_ = {class_label: n_samples}
                
                # Number of samples
                n_minority_samples = y_cluster_count[class_label]

                # Modify attributes for corner cases
                initial_attributes = self._modify_attributes(n_minority_samples, y_in_cluster.size)
                
                # Resample class data
                X_new_cluster, y_new_cluster = self._basic_sample(X_in_cluster, y_in_cluster)
                X_new_cluster, y_new_cluster = X_new_cluster[len(X_in_cluster):], y_new_cluster[len(X_in_cluster):]
                X_new, y_new = np.vstack((X_new, X_new_cluster)), np.hstack((y_new, y_new_cluster)) 

                # Restore modified attributes
                for attribute, value in initial_attributes.items():
                    setattr(self, attribute, value)
            
            # Append intracluster ratio
            self.intra_ratios_.append((label, self.ratio_.copy()))
        
        # Restore initial ratio
        self.ratio_ = initial_ratio

        return X_new, y_new

    def _inter_sample(self, X, y, initial_ratio):
        """Intercluster resampling."""

        # Random state
        random_state = check_random_state(self.random_state)

        # Initialize arrays of new data
        X_new = np.array([], dtype=X.dtype).reshape(0, X.shape[1])
        y_new = np.array([], dtype=y.dtype)

        # Number of nearest neighbors
        if hasattr(self, 'k_neighbors'):
            k = self.k_neighbors
        elif hasattr(self, 'n_neighbors'):
            k = self.n_neighbors
        else:
            return X_new, y_new

        # Intercluster oversampling
        self.inter_ratios_ = []
        for (label1, label2), proportion in self.distributor_.inter_distribution_:
            
            # Filter data in cluster 1 and cluster 2
            mask1, mask2 = (self.clusterer_.labels_ == label1), (self.clusterer_.labels_ == label2)
            X_in_cluster1, y_in_cluster1, X_in_cluster2, y_in_cluster2 = X[mask1], y[mask1], X[mask2], y[mask2]

            # Calculate ratio in the clusters
            clusters_ratio = {class_label: (round(n_samples * proportion) 
                              if class_label in y_in_cluster1 and class_label in y_in_cluster2 else 0)
                              for class_label, n_samples in initial_ratio.items()}

            # Resample data
            for class_label, n_samples in clusters_ratio.items():

                # Modify ratio
                self.ratio_ = {class_label: 1}

                for _ in range(n_samples):
                    
                    # Identify clusters
                    ind = random_state.choice([1, -1])
                    (X1, X2), (y1, y2) = [X_in_cluster1, X_in_cluster2][::ind], [y_in_cluster1, y_in_cluster2][::ind]
                    
                    # Select randomly a minority class sample from cluster 1
                    ind1 = random_state.choice(np.where(y1 == class_label)[0])
                    X1_class, y1_class = X1[ind1:(ind1 + 1)], y1[ind1:(ind1 + 1)]
                    
                    # Select minority class samples from cluster 2
                    ind2 = np.where(y2 == class_label)[0]
                    X2_class, y2_class = X2[ind2], y2[ind2] 

                    # Calculate distance matrix
                    X_class = np.vstack((X1_class, X2_class))
                    k_nn = min(k, len(X_class) - 1)
                    nn = check_neighbors_object('nn', k_nn).fit(X_class)
                    ind_nn = random_state.choice(nn.kneighbors()[1][0])

                    # Resample class data
                    X_in_clusters = np.vstack((X1_class, X2_class[(ind_nn - 1): ind_nn], X1[y1 != class_label], X2[y2 != class_label]))
                    y_in_clusters = np.hstack((y1_class, y2_class[(ind_nn - 1): ind_nn], y1[y1 != class_label], y2[y2 != class_label]))

                    # Modify attributes for corner cases
                    initial_attributes = self._modify_attributes(1, y_in_clusters.size)

                    # Resample class data
                    X_new_cluster, y_new_cluster = self._basic_sample(X_in_clusters, y_in_clusters)
                    X_new_cluster, y_new_cluster = X_new_cluster[len(X_in_clusters):], y_new_cluster[len(X_in_clusters):]
                    X_new, y_new = np.vstack((X_new, X_new_cluster)), np.hstack((y_new, y_new_cluster)) 

                    # Restore modified attributes
                    for attribute, value in initial_attributes.items():
                        setattr(self, attribute, value)

        # Restore initial ratio
        self.ratio_ = initial_ratio

        return X_new, y_new


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

        # Intracluster oversampling
        X_intra_new, y_intra_new = self._intra_sample(X, y, initial_ratio)
        
        # Intercluster oversampling
        X_inter_new, y_inter_new = self._inter_sample(X, y, initial_ratio)

        # Stack resampled data
        X_resampled, y_resampled = np.vstack((X, X_intra_new, X_inter_new)), np.hstack((y, y_intra_new, y_inter_new))

        return X_resampled, y_resampled
