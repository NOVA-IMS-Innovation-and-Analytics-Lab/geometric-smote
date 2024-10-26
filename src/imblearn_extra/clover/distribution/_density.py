"""Implementation of the DensityDistributor class."""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <fonsecajoao@protonmail.com>
# License: MIT

from __future__ import annotations

from collections import Counter
from itertools import product
from warnings import catch_warnings, filterwarnings

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_scalar
from typing_extensions import Self

from .. import InputData, Labels, Neighbors, Targets
from .base import BaseDistributor


class DensityDistributor(BaseDistributor):
    """Class to perform density based distribution.

    Samples are distributed based on the density of clusters.

    Read more in the [user_guide].

    Args:
        filtering_threshold:
            The threshold of a filtered cluster. It can be any non-negative number or
            `'auto'` to be calculated automatically.

            - If `'auto'`, the filtering threshold is calculated from the imbalance
            ratio of the target for the binary case or the maximum of the target's
            imbalance ratios for the multiclass case.

            - If `float` then it is manually set to this number. Any cluster that has an
            imbalance ratio smaller than the filtering threshold is identified as a filtered
            cluster and can be potentially used to generate minority class instances. Higher
            values increase the number of filtered clusters.

        distances_exponent:
            The exponent of the mean distance in the density calculation. It can be
            any non-negative number or `'auto'` to be calculated automatically.

            - If `'auto'` then it is set equal to the number of
            features. Higher values make the calculation of density more sensitive
            to the cluster's size i.e. clusters with large mean euclidean distance
            between samples are penalized.

            - If `float` then it is manually set to this number.

        sparsity_based:
            Whether sparse clusters receive more generated samples.

            - When `True` clusters receive generated samples that are inversely
            proportional to their density.

            - When `False` clusters receive generated samples that are proportional to their density.

        distribution_ratio:
            The ratio of intra-cluster to inter-cluster generated samples. It is a
            number in the `[0.0, 1.0]` range. The default value is `1.0`, a
            case corresponding to only intra-cluster generation. As the number
            decreases, less intra-cluster samples are generated. Inter-cluster
            generation, i.e. when `distribution_ratio` is less than `1.0`,
            requires a neighborhood structure for the clusters, i.e. a
            `neighbors_` attribute should be created after fitting and it will
            raise an error when it is not found.

    Attributes:
        clusters_density_ (Density):
            Each dict key is a multi-label tuple of shape `(cluster_label,
            class_label)`, while the values correspond to the density.

        distances_exponent_ (float):
            Actual exponent of the mean distance used in the calculations.

        distribution_ratio_ (float):
            A copy of the parameter in the constructor.

        filtered_clusters_ (List[MultiLabel]):
            Each element is a tuple of `(cluster_label, class_label)` pairs.

        filtering_threshold_ (float):
            Actual filtering threshold used in the calculations.

        inter_distribution_ (InterDistribution):
            Each dict key is a multi-label tuple of
            shape `((cluster_label1, cluster_label2), class_label)` while the
            values are the proportion of samples per class.

        intra_distribution_ (IntraDistribution):
            Each dict key is a multi-label tuple of shape `(cluster_label,
            class_label)` while the  values are the proportion of samples per class.

        labels_ (Labels):
            Labels of each sample.

        neighbors_ (Neighbors):
            An array that contains all neighboring pairs. Each row is
            a unique neighboring pair.

        majority_class_label_ (int):
            The majority class label.

        n_samples_ (int):
            The number of samples.

        sparsity_based_ (bool):
            A copy of the parameter in the constructor.

        unique_class_labels_ (Labels):
            An array of unique class labels.

        unique_cluster_labels_ (Labels):
            An array of unique cluster labels.

    Examples:
        >>> from clover.distribution import DensityDistributor
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.cluster import KMeans
        >>> from imblearn.datasets import make_imbalance
        >>> X, y = make_imbalance(
        ...     *load_iris(return_X_y=True),
        ...     sampling_strategy={0:50, 1:40, 2:30},
        ...     random_state=0
        ... )
        >>> labels = KMeans(random_state=0, n_init='auto').fit_predict(X, y)
        >>> density_distributor = DensityDistributor().fit(X, y, labels)
        >>> density_distributor.filtered_clusters_
        [(6, 1), (0, 1), (3, 1), (7, 1), (5, 2), (2, 2), (3, 2), (6, 2), (0, 2)]
        >>> density_distributor.intra_distribution_
        {(6, 1): 0.50604609281056... (0, 1): 0.143311766542165...}
        >>> density_distributor.inter_distribution_
        {}
    """

    def __init__(
        self: Self,
        filtering_threshold: float | str = 'auto',
        distances_exponent: float | str = 'auto',
        sparsity_based: bool = True,
        distribution_ratio: float = 1.0,
    ) -> None:
        self.filtering_threshold = filtering_threshold
        self.distances_exponent = distances_exponent
        self.sparsity_based = sparsity_based
        self.distribution_ratio = distribution_ratio

    def _check_parameters(
        self: Self,
        X: InputData,
        y: Targets,
        neighbors: Neighbors | None,
    ) -> Self:
        """Check distributor parameters."""

        # Filtering threshold
        if self.filtering_threshold == 'auto':
            counts_vals = Counter(y).values()
            self.filtering_threshold_ = max(counts_vals) / min(counts_vals)
        else:
            self.filtering_threshold_ = check_scalar(
                self.filtering_threshold,
                'filtering_threshold',
                (int, float),
                min_val=0.0,
            )

        # Distances exponent
        if self.distances_exponent == 'auto':
            self.distances_exponent_ = X.shape[1]
        else:
            self.distances_exponent_ = check_scalar(
                self.distances_exponent,
                'distances_exponent',
                (int, float),
                min_val=0.0,
            )

        # Sparsity based
        check_scalar(self.sparsity_based, 'sparsity_based', bool)
        self.sparsity_based_ = self.sparsity_based

        # distribution ratio
        check_scalar(
            self.distribution_ratio,
            'distribution_ratio',
            float,
            min_val=0.0,
            max_val=1.0,
        )
        max_distribution_ratio = 1.0
        if self.distribution_ratio < max_distribution_ratio and neighbors is None:
            msg = 'Parameter `distribution_ratio` should be equal to 1.0, when `neighbors` parameter is `None`.'
            raise ValueError(msg)
        self.distribution_ratio_ = self.distribution_ratio
        return self

    def _identify_filtered_clusters(self: Self, y: Targets) -> Self:
        """Identify the filtered clusters."""
        # Generate multi-label
        multi_labels = list(zip(self.labels_, y, strict=True))

        # Count multi-label
        multi_labels_counts = Counter(multi_labels)

        # Extract unique cluster and class labels
        unique_multi_labels = [
            multi_label for multi_label in multi_labels_counts if multi_label[1] not in self.majority_class_labels_
        ]

        # Identify filtered clusters
        self.filtered_clusters_ = []
        for multi_label in unique_multi_labels:
            n_minority_samples = multi_labels_counts[multi_label]
            n_majority_samples = multi_labels_counts[(multi_label[0], self.majority_class_labels_[0])]
            if n_majority_samples <= n_minority_samples * self.filtering_threshold_:
                self.filtered_clusters_.append(multi_label)

        return self

    def _calculate_clusters_density(self: Self, X: InputData, y: Targets) -> Self:
        """Calculate the density of the filtered clusters."""
        self.clusters_density_ = {}

        # Calculate density
        finite_densities = []
        for cluster_label, class_label in self.filtered_clusters_:
            # Calculate number of majority and minority samples in each cluster
            mask = (self.labels_ == cluster_label) & (y == class_label)
            n_minority_samples = mask.sum()

            # Calculate density
            n_minority_pairs = (n_minority_samples - 1) * n_minority_samples if n_minority_samples > 1 else 1
            mean_distances = euclidean_distances(X[mask]).sum() / n_minority_pairs
            with catch_warnings():
                filterwarnings('ignore')
                density = n_minority_samples / (mean_distances**self.distances_exponent_)
                if np.isfinite(density):
                    finite_densities.append(density)
                self.clusters_density_[(cluster_label, class_label)] = density

        # Convert zero and infinite densities
        min_density = 0.0
        if min_density in self.clusters_density_.values():
            self.clusters_density_ = {
                multi_label: 1.0 for multi_label, density in self.clusters_density_.items() if density == min_density
            }
            self.filtered_clusters_ = [
                multi_label for multi_label in self.filtered_clusters_ if multi_label in self.clusters_density_
            ]
        else:
            max_density = max(finite_densities) if finite_densities else 1.0
            self.clusters_density_ = {
                multi_label: (max_density if np.isinf(density) else density)
                for multi_label, density in self.clusters_density_.items()
            }
        return self

    def _intra_distribute(
        self: Self,
        X: InputData,
        y: Targets,
        labels: Labels | None,
        neighbors: Neighbors | None,
    ) -> Self:
        """In the clusters distribution.

        Distribute the generated samples in each cluster based on their
        density.
        """

        # Calculate weights based on density
        weights = {
            multi_label: (1 / density if self.sparsity_based_ else density)
            for multi_label, density in self.clusters_density_.items()
        }

        # Calculate normalization factors
        class_labels = {class_label for _, class_label in self.filtered_clusters_}
        normalization_factors = {class_label: 0.0 for class_label in class_labels}
        for (_, class_label), weight in weights.items():
            normalization_factors[class_label] += weight

        # Intra distribution
        self.intra_distribution_ = {
            multi_label: (self.distribution_ratio_ * weight / normalization_factors[multi_label[1]])
            for multi_label, weight in weights.items()
        }

        return self

    def _inter_distribute(
        self: Self,
        X: InputData,
        y: Targets,
        labels: Labels | None,
        neighbors: Neighbors | None,
    ) -> Self:
        """Between the clusters distribution.

        Distribute the generated samples between clusters based on their
        density.
        """

        # Identify filtered neighboring clusters
        filtered_neighbors = []
        class_labels = {class_label for _, class_label in self.filtered_clusters_}
        for pair, class_label in product(self.neighbors_, class_labels):
            multi_label0 = (pair[0], class_label)
            multi_label1 = (pair[1], class_label)
            if multi_label0 in self.filtered_clusters_ and multi_label1 in self.filtered_clusters_:
                filtered_neighbors.append((multi_label0, multi_label1))

        # Calculate inter-cluster density
        inter_clusters_density = {
            multi_labels: (self.clusters_density_[multi_labels[0]] + self.clusters_density_[multi_labels[1]])
            for multi_labels in filtered_neighbors
        }

        # Calculate weights based on density
        weights = {
            multi_labels: (1 / density if self.sparsity_based_ else density)
            for multi_labels, density in inter_clusters_density.items()
        }

        # Calculate normalization factors
        normalization_factors = {class_label: 0.0 for class_label in class_labels}
        for multi_labels, weight in weights.items():
            normalization_factors[multi_labels[0][1]] += weight

        # Intra distribution
        self.inter_distribution_ = {
            multi_labels: ((1 - self.distribution_ratio_) * weight / normalization_factors[multi_labels[0][1]])
            for multi_labels, weight in weights.items()
        }

        return self

    def _fit(
        self: Self,
        X: InputData,
        y: Targets,
        labels: Labels | None,
        neighbors: Neighbors | None,
    ) -> Self:
        # Check distributor parameters
        self._check_parameters(X, y, neighbors)

        # Identify filtered clusters
        self._identify_filtered_clusters(y)

        # Calculate density of filtered clusters
        self._calculate_clusters_density(X, y)

        super()._fit(X, y, labels, neighbors)

        return self
