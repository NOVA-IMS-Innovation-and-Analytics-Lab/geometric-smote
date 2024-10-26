"""Includes the implementation of KMeans-SMOTE."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_scalar
from typing_extensions import Self

from .. import InputData, Targets
from ..distribution._density import DensityDistributor
from ._cluster import ClusterOverSampler


class KMeansSMOTE(ClusterOverSampler):
    """KMeans-SMOTE algorithm.

    Applies KMeans clustering to the input space before applying SMOTE. Read
    more in the [user_guide].

    Args:
        sampling_strategy:
            Sampling information to resample the data set.

            - When `float`, it corresponds to the desired ratio of the number of
            samples in the minority class over the number of samples in the
            majority class after resampling. It is only available for binary
            classification.

            - When `str`, specify the class targeted by the resampling. The
            number of samples in the different classes will be equalized.
            Possible choices are:
                - `'minority'`: resample only the minority class.
                - `'not minority'`: resample all classes but the minority class.
                - `'not majority'`: resample all classes but the majority class.
                - `'all'`: resample all classes.
                - `'auto'`: equivalent to `'not majority'`.

            - When `dict`, the keys correspond to the targeted classes. The
            values correspond to the desired number of samples for each targeted
            class.

            - When callable, function taking `y` and returns a `dict`. The keys
            correspond to the targeted classes. The values correspond to the
            desired number of samples for each class.

        random_state:
            Control the randomization of the algorithm.

            - If `int`, it is the seed used by the random number
            generator.
            - If `np.random.RandomState` instance, it is the random number
            generator.
            - If `None`, the random number generator is the `RandomState`
            instance used by `np.random`.

        k_neighbors:
            Defines the number of nearest neighbors to be used by SMOTE.

            - If `int`, this number is used to construct synthetic
            samples.

            - If `object`, an estimator that inherits from
            `sklearn.neighbors.base.KNeighborsMixin` that will be
            used to find the number of nearest neighbors.

        kmeans_estimator:
            Defines the KMeans clusterer applied to the input space.

            - If `None`, `sklearn.cluster.MiniBatchKMeans` is used which
            tends to be better with large number of samples.

            - If KMeans object, then an instance from either
            `sklearn.cluster.KMeans` or `sklearn.cluster.MiniBatchKMeans`.

            - If `int`, the number of clusters to be used.

            - If `float`, the proportion of the number of clusters over the number
            of samples to be used.

        imbalance_ratio_threshold:
            The threshold of a filtered cluster. It can be any non-negative number or
            `'auto'` to be calculated automatically.

            - If `'auto'`, the filtering threshold is calculated from the imbalance
            ratio of the target for the binary case or the maximum of the target's
            imbalance ratios for the multiclass case.

            - If `float` then it is manually set to this number.

            Any cluster that has an imbalance ratio smaller than the filtering threshold is
            identified as a filtered cluster and can be potentially used to generate
            minority class instances. Higher values increase the number of filtered
            clusters.

        distances_exponent:
            The exponent of the mean distance in the density calculation. It can be
            any non-negative number or `'auto'` to be calculated automatically.

            - If `'auto'` then it is set equal to the number of
            features. Higher values make the calculation of density more sensitive
            to the cluster's size i.e. clusters with large mean euclidean distance
            between samples are penalized.

            - If `float` then it is manually set to this number.

        raise_error:
            Raise an error when no samples are generated.

            - If `True`, it raises an error when no filtered clusters are
            identified and therefore no samples are generated.

            - If `False`, it displays a warning.

        n_jobs:
            Number of CPU cores used.

            - If `None`, it means `1` unless in a `joblib.parallel_backend` context.

            - If `-1` means using all processors.

    Attributes:
        oversampler_ (imblearn.over_sampling.SMOTE):
            A fitted `imblearn.over_sampling.SMOTE` instance.

        clusterer_ (sklearn.cluster.KMeans | sklearn.cluster.MiniBatchKMeans):
            A fitted `sklearn.cluster.KMeans` or `sklearn.cluster.MiniBatchKMeans` instance.

        distributor_ (clover.distribution.DensityDistributor):
            A fitted `clover.distribution.DensityDistributor` instance.

        labels_ (Labels):
            Cluster labels of each sample.

        neighbors_ (None):
            It is `None` since KMeans does not support this attribute.

        random_state_ (np.random.RandomState):
            An instance of `np.random.RandomState` class.

        sampling_strategy_ (dict[int, int]):
            Actual sampling strategy.

    Examples:
        >>> import numpy as np
        >>> from clover.over_sampling import KMeansSMOTE
        >>> from sklearn.datasets import make_blobs
        >>> blobs = [100, 800, 100]
        >>> X, y  = make_blobs(blobs, centers=[(-10, 0), (0,0), (10, 0)])
        >>> # Add a single 0 sample in the middle blob
        >>> X = np.concatenate([X, [[0, 0]]])
        >>> y = np.append(y, 0)
        >>> # Make this a binary classification problem
        >>> y = y == 1
        >>> kmeans_smote = KMeansSMOTE(random_state=42)
        >>> X_res, y_res = kmeans_smote.fit_resample(X, y)
        >>> # Find the number of new samples in the middle blob
        >>> n_res_in_middle = ((X_res[:, 0] > -5) & (X_res[:, 0] < 5)).sum()
        >>> print("Samples in the middle blob: %s" % n_res_in_middle)
        Samples in the middle blob: 801
        >>> print("Middle blob unchanged: %s" % (n_res_in_middle == blobs[1] + 1))
        Middle blob unchanged: True
        >>> print("More 0 samples: %s" % ((y_res == 0).sum() > (y == 0).sum()))
        More 0 samples: True
    """

    def __init__(
        self: Self,
        sampling_strategy: dict[int, int] | str = 'auto',
        random_state: np.random.RandomState | int | None = None,
        k_neighbors: NearestNeighbors | int = 5,
        kmeans_estimator: KMeans | None = None,
        imbalance_ratio_threshold: float | str = 'auto',
        distances_exponent: float | str = 'auto',
        raise_error: bool = True,
        n_jobs: int | None = None,
    ) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.kmeans_estimator = kmeans_estimator
        self.imbalance_ratio_threshold = imbalance_ratio_threshold
        self.distances_exponent = distances_exponent
        self.raise_error = raise_error
        self.n_jobs = n_jobs

    def _check_estimators(self: Self, X: InputData, y: Targets) -> Self:
        """Check various estimators."""
        # Check oversampler
        self.oversampler_ = SMOTE(
            sampling_strategy=self.sampling_strategy,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state_,
            n_jobs=self.n_jobs,
        )

        # Check clusterer
        if self.kmeans_estimator is None:
            self.clusterer_ = MiniBatchKMeans(random_state=self.random_state_, n_init='auto')
        elif isinstance(self.kmeans_estimator, int):
            check_scalar(self.kmeans_estimator, 'kmeans_estimator', int, min_val=1)
            self.clusterer_ = MiniBatchKMeans(
                n_clusters=self.kmeans_estimator,
                random_state=self.random_state_,
                n_init='auto',
            )
        elif isinstance(self.kmeans_estimator, float):
            check_scalar(
                self.kmeans_estimator,
                'kmeans_estimator',
                float,
                min_val=0.0,
                max_val=1.0,
            )
            n_clusters = round((X.shape[0] - 1) * self.kmeans_estimator + 1)
            self.clusterer_ = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto')
        elif isinstance(self.kmeans_estimator, KMeans | MiniBatchKMeans):
            self.clusterer_ = clone(self.kmeans_estimator)
        else:
            msg = (
                'Parameter `kmeans_estimator` should be either `None` or the number of clusters '
                'or a float in the [0.0, 1.0] range equal to the number of clusters over the number '
                'of samples or an instance of either `KMeans` or `MiniBatchKMeans` class.'
            )
            raise TypeError(msg)

        # Check distributor
        self.distributor_ = DensityDistributor(
            filtering_threshold=self.imbalance_ratio_threshold,
            distances_exponent=self.distances_exponent,
        )

        return self
