"""
The :mod:`sklearnext.cluster._clusterers` extends standard clusterers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from sklearn import cluster


class KMeans(cluster.KMeans):

    def fit(self, X, y=None):
        
        # Modify number of clusters
        n_clusters = self.n_clusters
        if isinstance(self.n_clusters, float) and self.n_clusters <= 1.0:
            self.n_clusters_ = self.n_clusters = int(self.n_clusters * (len(X) - 1) + 1)

        # Call superclass method
        super(KMeans, self).fit(X, y)

        # Restore number of clusters
        self.set_params(n_clusters=n_clusters)
        
        return self
