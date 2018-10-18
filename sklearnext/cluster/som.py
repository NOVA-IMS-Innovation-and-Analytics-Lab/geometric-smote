"""
The :mod:`sklearnext.cluster.som` contains the 
implementation of Self-Organizing Map (SOM) clusterer.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from itertools import product

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array
from sklearn.preprocessing import minmax_scale
from somoclu import Somoclu


class SOM(BaseEstimator, ClusterMixin):
    """Class for training and visualizing a self-organizing map.

    Parameters
    ----------

    n_columns : int, default: 5
        The number of columns in the map.

    n_rows : int, default: 5
        The number of rows in the map.

    n_clusters : float, default: None
        The proportion of clusters relative to the number of samples of the input 
        space. If this is not None then `n_columns` and `n_rows` are ignored.

    initialcodebook : 2D numpy.array of float32 or None, default: None
        Define the codebook to start the training.

    kerneltype : int, default: 0
        Specify which kernel to use. 
        
        0 for dense CPU kernel.
        
        1 for dense GPU kernel if compiled with it.

    maptype : str, default: "planar" 
        Specify the map topology. 
        
        "planar" for planar map.
        
        "toroid" for toroid map.

    gridtype : str, default: "rectangular"
        Specify the grid form of the nodes. 
        
        "rectangular" for rectangular neurons.
        
        "hexagonal" for hexagonal neurons.

    compactsupport : bool, default: True 
        Cut off map updates beyond the training radius with the Gaussian neighborhood.
                           
    neighborhood : str, default: "gaussian" 
        Specify the neighborhood.
        
        "gaussian" for Gaussian neighborhood.
        
        "bubble" for bubble neighborhood function.

    std_coeff : float, default: 0.5
        Set the coefficient in the Gaussian neighborhood function exp(-||x-y||^2/(2*(coeff*radius)^2)).
    
    initialization : str or None, default: None 
        Specify the codebook initalization.
        
        "random" for random weights in the codebook.
        
        "pca": codebook is initialized from the first subspace spanned by the first 
        two eigenvectors of the correlation matrix.

    verbose : int, default: 0 
        Specify verbosity level (0, 1, or 2).
    """

    _attributes = ['train', 'codebook', 'bmus']

    def __init__(self, n_columns=5, n_rows=5, n_clusters=None, initialcodebook=None,
                 kerneltype=0, maptype="planar", gridtype="rectangular",
                 compactsupport=True, neighborhood="gaussian", std_coeff=0.5,
                 initialization=None, verbose=0):

        self.n_columns = n_columns
        self.n_rows = n_rows
        self.n_clusters = n_clusters
        self.initialcodebook = initialcodebook
        self.kerneltype = kerneltype
        self.maptype = maptype
        self.gridtype = gridtype
        self.compactsupport = compactsupport
        self.neighborhood = neighborhood
        self.std_coeff = std_coeff
        self.initialization = initialization
        self.verbose = verbose

    @staticmethod
    def _generate_labels_mapping(grid_labels):
        """Generate a mapping between grid labels and cluster labels."""

        # Identify unique grid labels
        unique_labels = [tuple(grid_label) for grid_label in np.unique(grid_labels, axis=0)]

        # Generate mapping
        labels_mapping = {grid_label: cluster_label for grid_label, cluster_label in  zip(unique_labels, range(len(unique_labels)))}

        return labels_mapping

    def _return_topological_neighbors(self, col, row):
        """Return the topological neighbors of a neuron."""
        
        # Return common topological neighbors for the two grid types
        topological_neighbors = [(col - 1, row), (col + 1, row), (col, row - 1), (col, row + 1)]
        
        # Append extra topological neighbors for hexagonal grid type
        if self.gridtype == 'hexagonal':
            offset = (-1) ** row
            topological_neighbors += [(col - offset, row - offset), (col - offset, row + offset)]

        # Apply constraints
        topological_neighbors = [(col, row) for col, row in topological_neighbors if 0 <= col < self.n_columns_ and 0 <= row < self.n_rows_ and [col, row] in self.algorithm_.bmus.tolist()]
        
        return topological_neighbors
    
    def _generate_neighbors(self, grid_labels, labels_mapping):
        """Generate pairs of neighboring labels."""

        # Generate grid topological neighbors
        grid_topological_neighbors = [product([grid_label], self._return_topological_neighbors(*grid_label)) for grid_label in grid_labels]

        # Flatten grid topological neighbors
        grid_topological_neighbors = [pair for pairs in grid_topological_neighbors for pair in pairs]

        # Generate cluster neighbors
        all_neighbors = [(labels_mapping[pair[0]], labels_mapping[pair[1]]) for pair in grid_topological_neighbors]
        all_neighbors = [tuple(pair) for pair in np.unique(all_neighbors, axis=0)]

        # Keep unique unordered pairs
        neighbors = []
        for pair in all_neighbors:
            if pair not in neighbors and pair[::-1] not in neighbors:
                neighbors.append(pair)
    
        return neighbors

    def fit(self, X, y=None, **fit_params):
        """Train the self-organizing map.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster.

        y : Ignored
        """

        # Check and normalize input data
        X = minmax_scale(check_array(X, dtype=np.float32))
        
        # Initialize Somoclu object
        if not hasattr(self, 'algorithm_'):

            # Set number of columns and rows from number of clusters
            if self.n_clusters is not None:
                self.n_columns_ = self.n_rows_ = int(self.n_clusters * (np.sqrt(len(X)) - 2) + 2)
            else:
                self.n_columns_, self.n_rows_ = self.n_columns, self.n_rows

            # Create object
            self.algorithm_ = Somoclu(n_columns=self.n_columns_, n_rows=self.n_rows_, initialcodebook=self.initialcodebook,
                                kerneltype=self.kerneltype, maptype=self.maptype, gridtype=self.gridtype,
                                compactsupport=self.compactsupport, neighborhood=self.neighborhood, 
                                std_coeff=self.std_coeff, initialization=self.initialization, data=None, 
                                verbose=self.verbose)
        
        # Fit Somoclu
        self.algorithm_.train(data=X, **fit_params)

        # Grid labels
        grid_labels = [tuple(grid_label) for grid_label in self.algorithm_.bmus]

        # Generate labels mapping
        labels_mapping = self._generate_labels_mapping(grid_labels)

        # Generate cluster labels
        self.labels_ = np.array([labels_mapping[grid_label] for grid_label in grid_labels])
        
        # Generate labels neighbors
        self.neighbors_ = self._generate_neighbors(grid_labels, labels_mapping)

        return self

    def fit_predict(self, X, y=None):
        """Train the self-organizing map and assign a cluster label to each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.

        u : Ignored

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X).labels_
