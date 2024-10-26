"""Base class for distributors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
#         Joao Fonseca <fonsecajoao@protonmail.com>
# License: MIT

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from typing_extensions import Self

from .. import InputData, InterDistribution, IntraDistribution, Labels, Neighbors, Targets


class BaseDistributor(BaseEstimator):
    """The base class for distributors.

    A distributor sets the proportion of samples to be generated inside
    each cluster and between clusters. Warning: This class should not be
    used directly. Use the derive classes instead.
    """

    def _intra_distribute(
        self: Self,
        X: InputData,
        y: Targets,
        labels: Labels | None,
        neighbors: Neighbors | None,
    ) -> Self:
        return self

    def _inter_distribute(
        self: Self,
        X: InputData,
        y: Targets,
        labels: Labels | None,
        neighbors: Neighbors | None,
    ) -> Self:
        return self

    def _validate_fitting(self: Self) -> Self:
        # Check labels
        if len(self.labels_) != self.n_samples_:
            msg = (
                f'Number of labels should be equal to the number of samples. '
                f'Got {len(self.labels_)} and {self.n_samples_} instead.'
            )
            raise ValueError(msg)

        # Check neighbors
        if not set(self.labels_).issuperset(self.neighbors_.flatten()):
            masg = 'Attribute `neighbors_` contains unknown labels.'
            raise ValueError(masg)
        unique_neighbors = {tuple(set(pair)) for pair in self.neighbors_}
        if len(unique_neighbors) < len(self.neighbors_):
            msg = 'Elements of `neighbors_` attribute are not unique.'
            raise ValueError(msg)

        # Check distribution
        proportions = {
            class_label: 0.0
            for class_label in self.unique_class_labels_
            if class_label not in self.majority_class_labels_
        }
        for (_, class_label), proportion in self.intra_distribution_.items():
            proportions[class_label] += proportion
        for (
            ((cluster_label1, class_label1), (cluster_label2, class_label2)),
            proportion,
        ) in self.inter_distribution_.items():
            if class_label1 != class_label2:
                multi_label = (
                    (cluster_label1, class_label1),
                    (cluster_label2, class_label2),
                )
                msg = (
                    'Multi-labels for neighboring cluster pairs should '
                    f'have a common class label. Got {multi_label} instead.'
                )
                raise ValueError(msg)
            proportions[class_label1] += proportion
        if not all(np.isclose(val, 0) or np.isclose(val, 1) for val in proportions.values()):
            msg = (
                'Intra-distribution and inter-distribution sum of proportions for each '
                f'class label should be either equal to 0 or 1. Got {proportions} instead.'
            )
            raise ValueError(msg)

        return self

    def _fit(
        self: Self,
        X: InputData,
        y: Targets,
        labels: Labels | None,
        neighbors: Neighbors | None,
    ) -> Self:
        if labels is not None:
            self._intra_distribute(X, y, labels, neighbors)
        if neighbors is not None:
            self._inter_distribute(X, y, labels, neighbors)
        return self

    def fit(
        self: Self,
        X: InputData,
        y: Targets,
        labels: Labels | None = None,
        neighbors: Neighbors | None = None,
    ) -> Self:
        """Generate the intra-label and inter-label distribution.

        Args:
            X:
                Matrix containing the data which have to be sampled.

            y:
                Corresponding label for each sample in X.
            labels:
                Labels of each sample.
            neighbors:
                An array that contains all neighboring pairs. Each row is
                a unique neighboring pair.

        Returns:
            The object itself.
        """
        # Check data
        X, y = check_X_y(X, y, dtype=None)

        # Set statistics
        counts = Counter(y)
        self.majority_class_labels_ = [
            class_label
            for class_label, class_label_count in counts.items()
            if class_label_count == max(counts.values())
        ]
        self.unique_cluster_labels_ = np.unique(labels) if labels is not None else np.array(0, dtype=int)
        self.unique_class_labels_ = np.unique(y)
        self.n_samples_ = len(X)

        # Set default attributes
        self.labels_ = np.repeat(0, len(X)) if labels is None else check_array(labels, ensure_2d=False)
        self.neighbors_ = np.empty((0, 2), dtype=int) if neighbors is None else check_array(neighbors, ensure_2d=False)
        self.intra_distribution_: IntraDistribution = {
            (0, class_label): 1.0 for class_label in np.unique(y) if class_label not in self.majority_class_labels_
        }
        self.inter_distribution_: InterDistribution = {}

        # Fit distributor
        self._fit(X, y, labels, neighbors)

        # Validate fitting procedure
        self._validate_fitting()

        return self

    def fit_distribute(
        self: Self,
        X: InputData,
        y: Targets,
        labels: Labels | None,
        neighbors: Neighbors | None,
    ) -> tuple[IntraDistribution, InterDistribution]:
        """Return the intra-label and inter-label distribution.

        Args:
            X:
                Matrix containing the data which have to be sampled.
            y:
                Corresponding label for each sample in X.
            labels:
                Labels of each sample.
            neighbors:
                An array that contains all neighboring pairs. Each row is
                a unique neighboring pair.

        Returns:
            distributions:
                A tuple with the two distributions.
        """
        self.fit(X, y, labels, neighbors)
        return self.intra_distribution_, self.inter_distribution_
