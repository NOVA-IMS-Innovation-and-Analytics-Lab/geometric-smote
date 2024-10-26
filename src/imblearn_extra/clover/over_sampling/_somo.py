"""Includes the implementation of SOMO."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

from math import sqrt

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_scalar
from typing_extensions import Self

from .. import InputData, Targets
from ..clusterer import SOM
from ..distribution._density import DensityDistributor
from ._cluster import ClusterOverSampler


class SOMO(ClusterOverSampler):
    """SOMO algorithm.

    Applies the SOM algorithm to the input space before applying SMOTE. Read
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

        som_estimator:
            Defines the SOM clusterer applied to the input space.

            - If `None`, `SOM` is used which
            tends to be better with large number of samples.

            - If SOM object, then it is a `clover.clusterer.SOM` instance.

            - If `int`, the number of clusters to be used.

            - If `float`, the proportion of the number of clusters over the number
            of samples to be used.

        distribution_ratio:
            The ratio of intra-cluster to inter-cluster generated samples. It is a
            number in the `[0.0, 1.0]` range. The default value is `0.8`, a
            number equal to the proportion of intra-cluster generated samples over
            the total number of generated samples. As the number decreases, less
            intra-cluster and more inter-cluster samples are generated.

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

        clusterer_ (clover.clusterer.SOM):
            A fitted `clover.clusterer.SOM` instance.

        distributor_ (clover.distribution.DensityDistributor):
            A fitted `clover.distribution.DensityDistributor` instance.

        labels_ (Labels):
            Cluster labels of each sample.

        neighbors_ (Neighbors):
            An array that contains all neighboring pairs with each row being
            a unique neighboring pair.

        random_state_ (np.random.RandomState):
            An instance of `np.random.RandomState` class.

        sampling_strategy_ (dict[int, int]):
            Actual sampling strategy.

    Examples:
        >>> import numpy as np
        >>> from clover.over_sampling import SOMO # doctest: +SKIP
        >>> from sklearn.datasets import make_blobs
        >>> blobs = [100, 800, 100]
        >>> X, y  = make_blobs(blobs, centers=[(-10, 0), (0,0), (10, 0)])
        >>> # Add a single 0 sample in the middle blob
        >>> X = np.concatenate([X, [[0, 0]]])
        >>> y = np.append(y, 0)
        >>> # Make this a binary classification problem
        >>> y = y == 1
        >>> somo = SOMO(random_state=42) # doctest: +SKIP
        >>> X_res, y_res = somo.fit_resample(X, y) # doctest: +SKIP
        >>> # Find the number of new samples in the middle blob
        >>> right, left = X_res[:, 0] > -5, X_res[:, 0] < 5 # doctest: +SKIP
        >>> n_res_in_middle = (right & left).sum() # doctest: +SKIP
        >>> print("Samples in the middle blob: %s" % n_res_in_middle) # doctest: +SKIP
        Samples in the middle blob: 801
        >>> unchanged = n_res_in_middle == blobs[1] + 1 # doctest: +SKIP
        >>> print("Middle blob unchanged: %s" % unchanged) # doctest: +SKIP
        Middle blob unchanged: True
        >>> more_zero_samples = (y_res == 0).sum() > (y == 0).sum() # doctest: +SKIP
        >>> print("More 0 samples: %s" % more_zero_samples) # doctest: +SKIP
        More 0 samples: True
    """

    def __init__(
        self: Self,
        sampling_strategy: dict[int, int] | str = 'auto',
        random_state: np.random.RandomState | int | None = None,
        k_neighbors: NearestNeighbors | int = 5,
        som_estimator: SOM | None = None,
        distribution_ratio: float = 0.8,
        raise_error: bool = True,
        n_jobs: int | None = None,
    ) -> None:
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.som_estimator = som_estimator
        self.distribution_ratio = distribution_ratio
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

        # Check clusterer and number of clusters
        if self.som_estimator is None:
            self.clusterer_ = SOM(random_state=self.random_state_)
        elif isinstance(self.som_estimator, int):
            check_scalar(self.som_estimator, 'som_estimator', int, min_val=1)
            n = round(sqrt(self.som_estimator))
            self.clusterer_ = SOM(n_columns=n, n_rows=n, random_state=self.random_state_)
        elif isinstance(self.som_estimator, float):
            check_scalar(self.som_estimator, 'som_estimator', float, min_val=0, max_val=1)
            n = round(sqrt((X.shape[0] - 1) * self.som_estimator + 1))
            self.clusterer_ = SOM(n_columns=n, n_rows=n, random_state=self.random_state_)
        elif isinstance(self.som_estimator, SOM):
            self.clusterer_ = clone(self.som_estimator)
        else:
            msg = (
                'Parameter `som_estimator` should be either `None` or the number of '
                'clusters or a float in the [0.0, 1.0] range equal to the number of '
                'clusters over the number of samples or an instance of the `SOM` class.'
            )
            raise TypeError(msg)

        # Check distributor
        self.distributor_ = DensityDistributor(
            distribution_ratio=self.distribution_ratio,
            filtering_threshold=1,
            distances_exponent=2,
        )

        return self
