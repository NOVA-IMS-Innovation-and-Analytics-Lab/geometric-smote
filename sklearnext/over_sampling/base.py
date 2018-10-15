"""
Extended base class for oversampling.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from abc import abstractmethod
from collections import Counter

import numpy as np
from sklearn.base import clone
from sklearn.utils import check_random_state
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_neighbors_object

from .distribution import DensityDistributor


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
