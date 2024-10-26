"""Implementation of the main class for clustering-based oversampling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from __future__ import annotations

import warnings
from collections import Counter, OrderedDict

import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.pipeline import Pipeline
from imblearn.utils import check_sampling_strategy
from imblearn.utils._validation import ArraysTransformer
from joblib import Parallel, delayed
from sklearn.base import ClusterMixin, TransformerMixin, clone
from sklearn.exceptions import FitFailedWarning
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from typing_extensions import Self

from .. import InputData, InterDistribution, IntraDistribution, Labels, Targets
from ..distribution import DensityDistributor
from ..distribution.base import BaseDistributor


def modify_nn(n_neighbors: NearestNeighbors | int, n_samples: int) -> NearestNeighbors | int:
    """Modify the nearest neighbors object.

    Args:
        n_neighbors:
            The `NearestNeighbors` object or number.
        n_samples:
            The number of samples.

    Returns:
        The modified `NearestNeighbors` object or number.
    """
    if isinstance(n_neighbors, NearestNeighbors):
        n_neighbors = (
            clone(n_neighbors).set_params(n_neighbors=n_samples - 1)
            if n_neighbors.n_neighbors >= n_samples
            else clone(n_neighbors)
        )
    elif isinstance(n_neighbors, int) and n_neighbors >= n_samples:
        n_neighbors = n_samples - 1
    return n_neighbors


def clone_modify(oversampler: BaseOverSampler, class_label: int, y_in_cluster: Targets) -> BaseOverSampler:
    """Clone and modify attributes of oversampler for corner cases.

    Args:
        oversampler:
            The oversampler to modify its attributes.
        class_label:
            The class label.
        y_in_cluster:
            The data of the target in the cluster.

    Returns:
        A cloned oversampler with modified number of nearest neighbors.
    """
    # Clone oversampler
    oversampler = clone(oversampler)

    # Not modify attributes case
    if isinstance(oversampler, RandomOverSampler):
        return oversampler

    # Select and modify oversampler
    n_minority_samples = Counter(y_in_cluster)[class_label]
    if n_minority_samples == 1:
        oversampler = RandomOverSampler()
    else:
        if hasattr(oversampler, 'k_neighbors'):
            oversampler.k_neighbors = modify_nn(oversampler.k_neighbors, n_minority_samples)
        if hasattr(oversampler, 'm_neighbors'):
            oversampler.m_neighbors = modify_nn(oversampler.m_neighbors, y_in_cluster.size)
        if hasattr(oversampler, 'n_neighbors'):
            oversampler.n_neighbors = modify_nn(oversampler.n_neighbors, n_minority_samples)
    return oversampler


def extract_intra_data(
    X: InputData,
    y: Targets,
    cluster_labels: Labels,
    intra_distribution: IntraDistribution,
    sampling_strategy: OrderedDict[int, int],
) -> list[tuple[dict[int, int], InputData, Targets]]:
    """Extract data for each filtered cluster.

    Args:
        X:
            The input data.
        y:
            The targets.
        cluster_labels:
            The cluster labels.
        intra_distribution:
            The intra-clusters distributions.
        sampling_strategy:
            The sampling strategy to follow.

    Returns:
        The intra-clusters data.
    """
    majority_class_label = Counter(y).most_common()[0][0]

    # Get offsets
    selected_multi_labels = []
    classes_labels = {class_label for _, class_label in intra_distribution}
    distribution_value_tie = 0.5
    for selected_class_label in classes_labels:
        intra_distribution_class_label = {
            (cluster_label, class_label): proportion
            for (cluster_label, class_label), proportion in intra_distribution.items()
            if class_label == selected_class_label
        }
        selected_multi_label = max(
            intra_distribution_class_label,
            key=lambda multi_label: intra_distribution_class_label[multi_label],
        )
        if intra_distribution_class_label[selected_multi_label] <= distribution_value_tie:
            selected_multi_labels.append(selected_multi_label)

    # Get clusters data
    clusters_data = []
    for (cluster_label, class_label), proportion in intra_distribution.items():
        mask = (cluster_labels == cluster_label) & (np.isin(y, [majority_class_label, class_label]))
        offset = int((cluster_label, class_label) in selected_multi_labels)
        n_minority_samples = int(round(sampling_strategy[class_label] * proportion)) + offset
        X_in_cluster, y_in_cluster = X[mask], y[mask]
        cluster_sampling_strategy = {class_label: n_minority_samples}
        if n_minority_samples > 0:
            clusters_data.append((cluster_sampling_strategy, X_in_cluster, y_in_cluster))
    return clusters_data


def extract_inter_data(
    X: InputData,
    y: Targets,
    cluster_labels: Labels,
    inter_distribution: InterDistribution,
    sampling_strategy: OrderedDict[int, int],
    random_state: np.random.RandomState,
) -> list[tuple[dict[int, int], InputData, Targets]]:
    """Extract data between filtered clusters.

    Args:
        X:
            The input data.
        y:
            The targets.
        cluster_labels:
            The cluster labels.
        inter_distribution:
            The inter-clusters distributions.
        sampling_strategy:
            The sampling strategy to follow.
        random_state:
            Control the randomization of the algorithm.

    Returns:
        The inter-clusters data.
    """
    majority_class_label = Counter(y).most_common()[0][0]
    clusters_data = []
    for (
        ((cluster_label1, class_label1), (cluster_label2, class_label2)),
        proportion,
    ) in inter_distribution.items():
        mask1 = (cluster_labels == cluster_label1) & (np.isin(y, [majority_class_label, class_label1]))
        mask2 = (cluster_labels == cluster_label2) & (np.isin(y, [majority_class_label, class_label2]))
        X1, X2, y1, y2 = X[mask1], X[mask2], y[mask1], y[mask2]
        majority_mask1, majority_mask2 = (
            (y1 == majority_class_label),
            (y2 == majority_class_label),
        )
        n_minority_samples = int(round(sampling_strategy[class_label1] * proportion))
        for _ in range(n_minority_samples):
            ind1, ind2 = (
                random_state.randint(0, (~majority_mask1).sum()),
                random_state.randint(0, (~majority_mask2).sum()),
            )
            X_in_clusters = np.vstack(
                (
                    X1[~majority_mask1][ind1].reshape(1, -1),
                    X2[~majority_mask2][ind2].reshape(1, -1),
                    X1[majority_mask1],
                    X2[majority_mask2],
                ),
            )
            y_in_clusters = np.hstack(
                (
                    y1[~majority_mask1][ind1],
                    y2[~majority_mask2][ind2],
                    y1[majority_mask1],
                    y2[majority_mask2],
                ),
            )
            clusters_sampling_strategy = {class_label1: 1}
            clusters_data.append((clusters_sampling_strategy, X_in_clusters, y_in_clusters))
    return clusters_data


def generate_in_cluster(
    oversampler: BaseOverSampler,
    transformer: TransformerMixin,
    cluster_sampling_strategy: dict[int, int],
    X_in_cluster: InputData,
    y_in_cluster: Targets,
) -> tuple[InputData, Targets]:
    """Generate intra-cluster or inter-cluster new samples.

    Args:
        oversampler:
            Oversampler to apply to each selected cluster.
        transformer:
            Transformer to apply before oversampling.
        cluster_sampling_strategy:
            The sampling strategy in the cluster.
        X_in_cluster:
            The input data in the cluster.
        y_in_cluster:
            The targets in the cluster.

    Returns:
        X_new:
            The generated.
        y_new:
            The corresponding label of resampled data.
    """

    # Create oversampler for specific cluster and class
    class_label = next(iter(cluster_sampling_strategy.keys()))
    oversampler = clone_modify(oversampler, class_label, y_in_cluster)
    oversampler.sampling_strategy_ = cluster_sampling_strategy
    oversampler.n_features_in_ = X_in_cluster.shape[1]

    # Resample cluster and class data
    X_res, y_res = oversampler._fit_resample(
        transformer.transform(X_in_cluster) if transformer is not None else X_in_cluster,
        y_in_cluster,
    )

    # Filter only new data
    X_new, y_new = X_res[len(X_in_cluster) :], y_res[len(y_in_cluster) :]

    return X_new, y_new


class ClusterOverSampler(BaseOverSampler):
    """A class that handles clustering-based oversampling.

    Any combination of oversampler, clusterer and distributor can
    be used.

    Read more in the [user_guide].

    Args:
        oversampler:
            Oversampler to apply to each selected cluster.

        clusterer:
            Clusterer to apply to input space before oversampling.

            - When `None`, it corresponds to a clusterer that assigns
            a single cluster to all the samples equivalent to no clustering.

            - When clusterer is given, it applies clustering to the input space. Then
            oversampling is applied inside each cluster and between clusters.

        distributor:
            Distributor to distribute the generated samples per cluster label.

            - When `None` and a clusterer is provided then it corresponds to the
            density distributor. If clusterer is also `None` than the distributor
            does not affect the over-sampling procedure.

            - When distributor object is provided, it is used to distribute the
            generated samples to the clusters.

        raise_error:
            Raise an error when no samples are generated.

            - If `True`, it raises an error when no filtered clusters are
            identified and therefore no samples are generated.

            - If `False`, it displays a warning.

        random_state:
            Control the randomization of the algorithm.

            - If `int`, it is the seed used by the random number
            generator.
            - If `np.random.RandomState` instance, it is the random number
            generator.
            - If `None`, the random number generator is the `RandomState`
            instance used by `np.random`.

        n_jobs:
            Number of CPU cores used.

            - If `None`, it means `1` unless in a `joblib.parallel_backend` context.

            - If `-1` means using all processors.

    Attributes:
        oversampler_ (imblearn.over_sampling.base.BaseOverSampler):
            A fitted clone of the `oversampler` parameter.

        clusterer_ (sklearn.base.ClusterMixin):
            A fitted clone of the `clusterer` parameter or `None` when a
            clusterer is not given.

        distributor_ (clover.distribution.base.BaseDistributor):
            A fitted clone of the `distributor` parameter or a fitted instance of
            the `DensityDistributor` when a distributor is not given.

        labels_ (Labels):
            Cluster labels of each sample.

        neighbors_ (Neighbors):
            An array that contains all neighboring pairs with each row being
            a unique neighboring pair. It is `None` when the clusterer does not
            support this attribute.

        random_state_ (np.random.RandomState):
            An instance of `np.random.RandomState` class.

        sampling_strategy_ (dict[int, int]):
            Actual sampling strategy.

    Examples:
        >>> from collections import Counter
        >>> from clover.over_sampling import ClusterOverSampler
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.cluster import KMeans
        >>> from imblearn.over_sampling import SMOTE
        >>> X, y = make_classification(random_state=0, n_classes=2, weights=[0.9, 0.1])
        >>> print('Original dataset shape %s' % Counter(y))
        Original dataset shape Counter({{0: 90, 1: 10}})
        >>> cluster_oversampler = ClusterOverSampler(
        ... oversampler=SMOTE(random_state=5),
        ... clusterer=KMeans(random_state=10, n_init='auto'))
        >>> X_res, y_res = cluster_oversampler.fit_resample(X, y)
        >>> print('Resampled dataset shape %s' % Counter(y_res))
        Resampled dataset shape Counter({{0: 90, 1: 90}})
    """

    def __init__(
        self: Self,
        oversampler: BaseOverSampler,
        clusterer: ClusterMixin | None = None,
        distributor: BaseDistributor | None = None,
        raise_error: bool = True,
        random_state: np.random.RandomState | int | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.oversampler = oversampler
        self.clusterer = clusterer
        self.distributor = distributor
        self.raise_error = raise_error
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self: Self, X: InputData, y: Targets) -> Self:
        """Check inputs and statistics of the sampler.

        You should use `fit_resample` to generate the synthetic data.

        Args:
            X:
                Data array.
            y:
                Target array.

        Returns:
            self:
                Return the instance itself.
        """
        X, y, _ = self._check_X_y(X, y)
        self._check(X, y)
        return self

    def fit_resample(
        self: Self,
        X: InputData,
        y: Targets,
        **fit_params: dict[str, str],
    ) -> tuple[InputData, Targets]:
        """Resample the dataset.

        Args:
            X:
                Matrix containing the data which have to be sampled.
            y:
                Corresponding label for each sample in X.
            fit_params:
                Parameters passed to the fit method of the clusterer.

        Returns:
            X_resampled:
                The array containing the resampled data.
            y_resampled:
                The corresponding label of resampled data.
        """
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, binarize_y = self._check_X_y(X, y)

        self._check(X, y)._fit(X, y, **fit_params)

        output = self._fit_resample(X, y)

        y_ = label_binarize(y=output[1], classes=np.unique(y)) if binarize_y else output[1]

        X_, y_ = arrays_transformer.transform(output[0], y_)
        return (X_, y_)

    def _cluster_sample(
        self: Self,
        clusters_data: list[tuple[dict[int, int], InputData, Targets]],
        X: InputData,
        y: Targets,
    ) -> tuple[InputData, Targets] | None:
        generated_data = Parallel(n_jobs=self.n_jobs)(
            delayed(generate_in_cluster)(self.oversampler_, self.transformer_, *data) for data in clusters_data
        )
        if generated_data:
            X, y = (np.concatenate(data) for data in zip(*generated_data, strict=True))
            return X, y
        return None

    def _intra_sample(self: Self, X: InputData, y: Targets) -> tuple[InputData, Targets] | None:
        clusters_data = extract_intra_data(
            X,
            y,
            self.labels_,
            self.distributor_.intra_distribution_,
            self.sampling_strategy_,
        )
        return self._cluster_sample(clusters_data, X, y)

    def _inter_sample(self: Self, X: InputData, y: Targets) -> tuple[InputData, Targets] | None:
        clusters_data = extract_inter_data(
            X,
            y,
            self.labels_,
            self.distributor_.inter_distribution_,
            self.sampling_strategy_,
            self.random_state_,
        )
        return self._cluster_sample(clusters_data, X, y)

    def _check_estimators(self: Self, X: InputData, y: Targets) -> Self:
        # Check transformer and oversampler
        if isinstance(self.oversampler, Pipeline):
            if self.oversampler.steps[:-1]:
                self.transformer_ = Pipeline(self.oversampler.steps[:-1]).fit(X)
            self.oversampler_ = clone(self.oversampler.steps[-1][-1])
        else:
            self.oversampler_ = clone(self.oversampler)

        # Check clusterer and distributor
        if self.clusterer is None and self.distributor is not None:
            msg = (
                'Distributor was found but clusterer is set to `None`. '
                'Either set parameter `distributor` to `None` or use a clusterer.'
            )
            raise ValueError(msg)
        elif self.clusterer is None and self.distributor is None:
            self.clusterer_ = None
            self.distributor_ = BaseDistributor()
        else:
            self.clusterer_ = clone(self.clusterer)
            self.distributor_ = DensityDistributor() if self.distributor is None else clone(self.distributor)
        return self

    def _check_sampling_strategy(self: Self, y: Targets) -> Self:
        self.sampling_strategy_ = check_sampling_strategy(
            self.oversampler_.sampling_strategy,
            y,
            self._sampling_type,
        )
        return self

    def _check(self: Self, X: InputData, y: Targets) -> Self:
        # Check random state
        self.random_state_ = check_random_state(self.random_state)

        # Check transformer
        self.transformer_ = None

        # Check estimators and sampling strategy
        self._check_estimators(X, y)._check_sampling_strategy(y)

        return self

    def _fit(self: Self, X: InputData, y: Targets, **fit_params: dict[str, str]) -> Self:
        # Fit clusterer
        if self.clusterer_ is not None:
            self.clusterer_.fit(X, y, **fit_params)

        # Extract labels and neighbors
        self.labels_ = getattr(self.clusterer_, 'labels_', np.zeros(len(X), dtype=int))
        self.neighbors_ = getattr(self.clusterer_, 'neighbors_', None)

        # fit distributor
        self.distributor_.fit(X, y, labels=self.labels_, neighbors=self.neighbors_)

        # Case when no samples are generated
        if not self.distributor_.intra_distribution_ and not self.distributor_.inter_distribution_:
            msg = 'No samples were generated. Try to modify the parameters of the clusterer or distributor.'

            # Raise error
            if self.raise_error:
                raise ValueError(msg)

            # Display warning
            else:
                warnings.warn(msg, FitFailedWarning, stacklevel=1)

        return self

    def _fit_resample(
        self: Self,
        X: InputData,
        y: Targets,
        **fit_params: dict[str, str],
    ) -> tuple[InputData, Targets]:
        # Intracluster oversampling
        data_intra = self._intra_sample(X, y)
        if data_intra is not None:
            X_intra_new, y_intra_new = data_intra
        else:
            X_intra_new, y_intra_new = None, None
        intra_count: Counter = Counter(y_intra_new)

        # Intercluster oversampling
        data_inter = self._inter_sample(X, y)
        if data_inter is not None:
            X_inter_new, y_inter_new = data_inter
        else:
            X_inter_new, y_inter_new = None, None
        inter_count: Counter = Counter(y_inter_new)

        # Set sampling strategy
        self.sampling_strategy_ = OrderedDict({})
        for class_label in set(intra_count.keys()).union(inter_count.keys()):
            self.sampling_strategy_[class_label] = intra_count.get(class_label, 0) + inter_count.get(class_label, 0)

        # Stack resampled data
        X_resampled_unstacked = [
            self.transformer_.transform(X) if self.transformer_ is not None else X,
            X_intra_new,
            X_inter_new,
        ]
        y_resampled_unstacked = [y, y_intra_new, y_inter_new]
        X_resampled, y_resampled = (
            np.vstack([X for X in X_resampled_unstacked if X is not None]),
            np.hstack([y for y in y_resampled_unstacked if y is not None]),
        )

        return X_resampled, y_resampled
