"""
Class to perform Self-Organizing Map Oversampling.
"""

from imblearn.over_sampling.base import BaseOverSampler
import somoclu as somo_algorithm
from collections import Counter
import numpy as np
from random import *
import pandas as pd
from sklearn.preprocessing import normalize
from math import isnan
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class SOMO(BaseOverSampler):
    """
    Self Organizing Map Oversampling Algorithm
    An oversampling algorithm that leverages the topological
    clustering characteristics of Self-Organizing Maps.
    Parameters
    ----------
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.
        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.
    som_rows : number of rows for two-dimensional output map of SOM
    som_cols : number of columns for two-dimensional output map of SOM
    iterations : number of iterations to train the Self Organizing Map
    filtered_cluster_ratio : Defined the treshold to identify a cluster as
        filtered cluster
    inter_intra_cluster_ratio : Describes the ratio of inter- and intracluster
        generated samples, a higher value indicates more intracluster created samples.
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 som_rows=10,
                 som_cols=10,
                 iterations=1000,
                 filtered_cluster_ratio=0.5,
                 inter_intra_cluster_ratio=0.5):
        super(SOMO, self).__init__(ratio=ratio, random_state=random_state)
        self.som_rows = som_rows
        self.som_cols = som_cols
        self.iterations = iterations
        self.filtered_cluster_ratio = filtered_cluster_ratio
        self.inter_intra_cluster_ratio = inter_intra_cluster_ratio

    def _cluster(self, X_sub):
        """
        Description: Clusters the normalized input data
        Returns ----------
        som - Object with topological cluster map
        """
        som = somo_algorithm.Somoclu(self.som_cols, self.som_rows, compactsupport=False)
        som.train(np.float32(X_sub), epochs=self.iterations)

        return som

    def _filter_Cluster(self, som, y_sub, class_sample):
        """
        Description: Identifies all filtered cluster and the number of minority
            samples belonging to them
        Returns ----------
        som_cluster : Array of assigned clusters for each data sample,
        filtered_cluster_size : array of filtered cluster and number of
            samples in each
        """

        som_cluster = pd.DataFrame(
            [int(''.join(map(str, num))) for num in som.bmus])
        som_cluster['Target'] = y_sub

        # Cluster ratio is not correctly ordered
        filtered_cluster_size = pd.DataFrame({'cluster':[], 'cluster_ratio':[], 'count':[]}, dtype=int)
        for clust in set(som_cluster[0]):
            temp = som_cluster[som_cluster[0] == clust]
            filtered_cluster_size = filtered_cluster_size.append({'cluster': clust,'cluster_ratio': (len(temp[temp['Target'] == class_sample]) / len(temp)), 'count': (len(temp[temp['Target'] == class_sample]))} ,ignore_index=True)

        filtered_cluster_size = filtered_cluster_size[filtered_cluster_size['cluster_ratio']>self.filtered_cluster_ratio]
        filtered_cluster_size = filtered_cluster_size.set_index('cluster')
        filtered_cluster_size.index.rename('', inplace=True)
        filtered_cluster_size = filtered_cluster_size.drop('cluster_ratio', axis=1)
        filtered_cluster_size = filtered_cluster_size[filtered_cluster_size > 1].dropna()

        return som_cluster, filtered_cluster_size

    def _calc_Distances(self, X_sub, class_sample, som_cluster,
                        filtered_cluster_size):
        """
        Description: Calculates average euclidean distances in each filtered
            cluster for each sample towards all other samples and averages them
        Returns ----------
        eucl_distances : array of average euclidean distance for each
            filtered cluster
        """
        eucl_distances = []
        for num in filtered_cluster_size.index:
            input_idx = som_cluster[
                (som_cluster[0] == num)
                & (som_cluster['Target'] == class_sample)].index
            individual_Distance = []
            for idx in input_idx:
                individual_Distance.append(
                    np.mean([(np.linalg.norm(X_sub[idx] - X_sub[num]))
                             for num in input_idx if num != idx]))
            eucl_distances.append(np.mean(individual_Distance))

        return eucl_distances

    def _calc_Density(self, filtered_cluster_size, eucl_distances):
        """
        Description: Calculates the density for each filtered cluster and neighbor
            relation
        Returns ----------
        filtered_cluster_size : DataFrame with density for each filtered cluster
        neighbors : 2d array with all filtered cluster relations and their density
        """
        # Calculate density in filtered cluster
        filtered_cluster_size['Density'] = np.divide(
            np.array(filtered_cluster_size, dtype=int).ravel(),
            np.square(eucl_distances))

        # Identify neighbors and calc Density
        neighbors = []
        fil_cl = pd.DataFrame(filtered_cluster_size.index.values).apply(
            pd.Series)
        for idx in fil_cl[0]:
            cur_neighbors = fil_cl[
                ((fil_cl <= idx + 1) & (fil_cl >= idx - 1))
                | ((fil_cl >= idx - 11) & (fil_cl <= idx - 9))
                | ((fil_cl <= idx + 11) & (fil_cl >= idx + 9))]
            for i in cur_neighbors[0]:
                if ((idx != i) & (isnan(i) != True)):
                    neighbors.append([idx, int(i)])
        for index, clust in pd.DataFrame(neighbors).iterrows():
            neighbors[index].append(
                (filtered_cluster_size.loc[clust[0]].Density +
                 filtered_cluster_size.loc[clust[1]].Density))

        return filtered_cluster_size, neighbors

    def _oversample_intra(self, intra_Samples, X_sub, y_sub, class_sample,
                          som_cluster, filtered_cluster_size):
        """
        Description: Oversamples minority class in each filtered cluster
        Returns ----------
        modified_X : oversampled X values
        modified_y : y values of minority class
        """
        modified_X = np.empty((0, len(X_sub[0])), int)
        modified_y = []
        # Calculate Weights
        weights = [(1 / den) / sum((1 / filtered_cluster_size.Density))
                   for den in filtered_cluster_size.Density]

        # Calculate amount of samples for each filtered cluster
        samples = [int(weight * intra_Samples) for weight in weights]
        while (intra_Samples - sum(samples) != 0):
            random_index = randrange(0, len(samples))
            samples[random_index] = samples[random_index] + 1

        # Oversample amount of samples
        for index, num in pd.DataFrame(filtered_cluster_size.index).iterrows():
            if samples[index] == 0:
                pass
            else:
                input_idx = som_cluster[(som_cluster[0] == num[0])].index
                cur_X, cur_y = X_sub[input_idx], y_sub[input_idx]
                if cur_y.mean() == class_sample:

                    rand_maj = choice(
                        np.where(y_sub == self.majority_class)[0])
                    cur_X, cur_y = np.append(
                        cur_X, [X_sub[rand_maj]], axis=0), np.append(
                            cur_y, y_sub[rand_maj])

                if len(cur_y[cur_y == class_sample]) <= 5:
                    k_neighbors = 1
                else:
                    k_neighbors = 5
                sm = SMOTE(
                    ratio={class_sample: (samples[index] + len(cur_X) - 1)},
                    k_neighbors=k_neighbors)
                over_X, over_y = sm.fit_sample(
                    np.array(cur_X), np.array(cur_y))
                over_X, over_y = over_X[-samples[index]:], over_y[
                    -samples[index]:]
                modified_X, modified_y = np.append(
                    modified_X, over_X, axis=0), np.append(modified_y, over_y)

        return modified_X, modified_y

    def _oversample_inter(self, inter_Samples, X_sub, y_sub, class_sample,
                          som_cluster, neighbors):
        """
        Description: Oversamples between filtered clusters that are topological neighbors
        Returns ----------
        modified_X : oversampled X values
        modified_y : y values of minority class
        """
        modified_X = np.empty((0, len(X_sub[0])), int)
        modified_y = []

        #Calculate Weights
        neigh = pd.DataFrame(neighbors)
        weights = [(1 / den) / sum((1 / neigh[2])) for den in neigh[2]]

        #Calculate Samples
        samples = [int(weight * inter_Samples) for weight in weights]
        # Check that all intra_Samples are spread
        while (inter_Samples - sum(samples) != 0):
            random_index = randrange(0, len(samples))
            samples[random_index] = samples[random_index] + 1

        # Oversample amount of samples
        for index, neigh in pd.DataFrame(neighbors).iterrows():
            for i in range(0, samples[index]):
                random_sample_A = choice(som_cluster[
                    (som_cluster[0] == neigh[0])
                    & (som_cluster['Target'] == class_sample)].index)
                random_sample_B = choice(som_cluster[
                    (som_cluster[0] == neigh[1])
                    & (som_cluster['Target'] == class_sample)].index)
                random_maj = choice(som_cluster[(
                    som_cluster['Target'] == self.majority_class)].index)
                cur_X, cur_y = X_sub[[
                    random_sample_A, random_sample_B, random_maj
                ]], y_sub[[random_sample_A, random_sample_B, random_maj]]
                sm = SMOTE(ratio={class_sample: 3}, k_neighbors=1)
                over_X, over_y = sm.fit_sample(
                    np.array(cur_X), np.array(cur_y))
                modified_X, modified_y = np.append(
                    modified_X, over_X[len(cur_X):], axis=0), np.append(
                        modified_y, over_y[len(cur_X):])
        return modified_X, modified_y

    def _sample(self, X, y):
        """
        Description: Oversampling of each minority class
        Returns ----------
        X - Original data input with oversampled data samples
        y - original input with oversampled target classes
        """
        X = normalize(X)
        self.majority_class = max(Counter(y), key=Counter(y).get)
        samples_X, samples_y = np.empty((0, len(X[0])), int), []

        som = self._cluster(X)

        for class_sample, num_samples in self.ratio_.items():

            if class_sample == self.majority_class:
                pass
            else:
                try:
                    som_cluster, filtered_cluster_size = self._filter_Cluster(
                        som, y, class_sample)

                    eucl_distances = self._calc_Distances(
                        X, class_sample, som_cluster, filtered_cluster_size)
                    filtered_cluster_size, neighbors = self._calc_Density(
                        filtered_cluster_size, eucl_distances)
                # Case that there are no neighbors
                    if (len(neighbors)> 0):

                        intra_X, intra_y = self._oversample_intra(
                            int(num_samples * self.inter_intra_cluster_ratio), X,
                            y, class_sample, som_cluster, filtered_cluster_size)
                        inter_X, inter_y = self._oversample_inter(
                            int(num_samples * (1 - self.inter_intra_cluster_ratio)),
                            X, y, class_sample, som_cluster, neighbors)
                        samples_X = np.append(
                            samples_X, np.append(intra_X, inter_X, axis=0), axis=0)
                        samples_y = np.append(samples_y, np.append(intra_y, inter_y))

                    else:
                        intra_X, intra_y = self._oversample_intra(
                        int(num_samples), X,
                            y, class_sample, som_cluster, filtered_cluster_size)
                        samples_X = np.append(samples_X,intra_X, axis=0)
                        samples_y = np.append(samples_y,intra_y)

                except ValueError:
                    print('No filtered Cluster were identified for class %s' %(class_sample))

        X = np.append(X, samples_X, axis=0)
        y = np.append(y, samples_y)

        return (X, y)
