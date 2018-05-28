from imblearn.over_sampling.base import BaseOverSampler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.spatial.distance import pdist
from imblearn.over_sampling import SMOTE
from warnings import filterwarnings, catch_warnings, warn
from sklearn.exceptions import DataConversionWarning
from math import isnan


class DBSCANSMOTE(BaseOverSampler):
    ''' Clusters the input data using DBScan and then oversamples using smote the defined clusters'''

    def __init__(self,
                 ratio="auto",
                 random_state=None,
                 normalize=True,
                 n_clusters = 8,
                 k_neighbors = 5,
                 n_jobs=1):

        super(DBSCANSMOTE, self).__init__(ratio=ratio, random_state=random_state)
        self._normalize = normalize
        self.n_clusters = n_clusters

        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _fit_cluster(self, X, y=None):
        """
        Normalises the data fits a dbscan cluster object
        :param X:
        :param y:
        :return:
        """

        if self._normalize:
            min_max = MinMaxScaler()
            # When the input data is int it will give a warning when converting to double
            with catch_warnings():
                filterwarnings("ignore", category=DataConversionWarning)
                X_ = min_max.fit_transform(X)
        else:
            X_ = X

        self._cluster_class.fit(X_, y)

    def _filter_clusters(self, y, cluster_labels=None, minority_label=None):
        """
        Calculates the clusters where the minority labels is dominant and returns them
        :param y:
        :param cluster_labels:
        :param minority_label:
        :return:
        """

        if cluster_labels is None:
            cluster_labels = self.labels

        unique_labels = np.unique(cluster_labels)

        # Remove label of observations identified as noise by DBSCAN:
        unique_labels = unique_labels[unique_labels != -1]

        filtered_clusters = []

        for label in unique_labels:
            cluster_obs = y[cluster_labels == label]

            minority_obs = cluster_obs[cluster_obs == minority_label].size
            majority_obs = cluster_obs[cluster_obs != minority_label].size

            imb_ratio = (majority_obs + 1) / (minority_obs + 1)

            if imb_ratio < 1:
                filtered_clusters.append(label)

        return filtered_clusters

    def _calculate_sampling_weights(self, X, y, filtered_clusters, cluster_labels=None, minority_class=None):

        if cluster_labels is None:
            cluster_labels = self.labels

        sparsity_factors = {}

        for cluster in filtered_clusters:

            # Observations belonging to current cluster and from the minority class
            obs = np.all([cluster_labels == cluster, y == minority_class], axis=0)
            n_obs = obs.sum()

            cluster_X = X[obs]

            # pdist calculates the condensed distance matrix, which is the upper triangle of the regular distance matrix
            # We can just calculate the mean over that vector, considering that that d(a,b) only exists once ( d(b,a) and
            # d(a,a) is not present).
            distance = pdist(cluster_X, 'euclidean')

            with catch_warnings():
                filterwarnings("ignore", category=RuntimeWarning, module="numpy")
                average_minority_distance = np.mean(distance)

            density_factor = average_minority_distance / (n_obs**2)

            sparsity_factor = 1 / density_factor

            sparsity_factors[cluster] = sparsity_factor

        sparsity_sum = sum(sparsity_factors.values())

        sampling_weights = {}

        for cluster in sparsity_factors:
            sampling_weights[cluster] = sparsity_factors[cluster] / sparsity_sum

        return sampling_weights

    def _sample(self, X, y):

        # Create the clusters and set the labels
        self._set_cluster()
        self._fit_cluster(X, y)

        self.labels = self._cluster_class.labels_

        X_resampled = X.copy()
        y_resampled = y.copy()

        with catch_warnings():
            filterwarnings("ignore", category=UserWarning, module="imblearn")

            for target_class in self.ratio_:

                n_to_generate = self.ratio_[target_class]

                clusters_to_use = self._filter_clusters(y, self._cluster_class.labels_, target_class)

                # In case we do not have cluster where the target class it dominant, we apply regular SMOTE
                if not clusters_to_use and n_to_generate > 0:
                    w = "Class %s does not have a cluster where is dominant." %(target_class)
                    warn(w)
                else:
                    sampling_weights = self._calculate_sampling_weights(X, y, clusters_to_use, self.labels, target_class)

                    for cluster in sampling_weights:
                        mask = self.labels == cluster
                        X_cluster = X[mask]
                        y_cluster = y[mask]

                        n_obs = mask.sum()

                        artificial_index = -1

                        # There needs to be at least two unique values of the target variable
                        if np.unique(y_cluster).size < 2:
                            art_x = np.zeros((1, X.shape[1]))
                            artificial_index = n_obs

                            artificial_y = np.unique(y)[np.unique(y) != target_class][0]

                            X_cluster = np.concatenate((X_cluster, art_x), axis=0)
                            y_cluster = np.concatenate((y_cluster, np.asarray(artificial_y).reshape((1,))), axis=0)

                        minority_obs = y_cluster[y_cluster == target_class]

                        n_new = n_to_generate * sampling_weights[cluster]

                        if isnan(n_new):
                            n_new = 0

                        temp_dic = {target_class: int(round(n_new) + minority_obs.size)}

                        # We need to make sure that k_neighors is less than the number of observations in the cluster
                        if self.k_neighbors > minority_obs.size - 1:
                            k_neighbors = minority_obs.size - 1
                        else:
                            k_neighbors = self.k_neighbors

                        over_sampler = SMOTE(ratio=temp_dic, k_neighbors=k_neighbors)
                        over_sampler.fit(X_cluster, y_cluster)

                        X_cluster_resampled, y_cluster_resampled = over_sampler.sample(X_cluster, y_cluster)

                        # If there was a observation added, then it is necessary to remove it now
                        if artificial_index > 0:
                            X_cluster_resampled = np.delete(X_cluster_resampled, artificial_index, axis=0)
                            y_cluster_resampled = np.delete(y_cluster_resampled, artificial_index)

                        # Save the newly generated samples only
                        X_cluster_resampled = X_cluster_resampled[n_obs:, :]
                        y_cluster_resampled = y_cluster_resampled[n_obs:, ]

                        # Add the newly generated samples to the data to be returned
                        X_resampled = np.concatenate((X_resampled, X_cluster_resampled))
                        y_resampled = np.concatenate((y_resampled, y_cluster_resampled))

        return X_resampled, y_resampled

    def get_labels(self):
        """Returns the labels of the data points"""
        return self._cluster_class.labels_

    def _set_cluster(self):
        self._cluster_class = KMeans(n_clusters= self.n_clusters)
