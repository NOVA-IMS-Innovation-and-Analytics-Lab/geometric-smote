"""Implementation of the Self-Organizing Map (SOM) clusterer."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from collections.abc import Callable
from itertools import product
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from minisom import MiniSom, asymptotic_decay
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils import check_array, check_random_state
from typing_extensions import Self


def generate_labels_mapping(labels_coords: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Generate a mapping between grid labels and cluster labels."""

    # Identify unique grid labels
    unique_labels = sorted(set(labels_coords))

    # Generate mapping
    labels_mapping = dict(zip(unique_labels, range(len(unique_labels)), strict=True))

    return labels_mapping


def extract_topological_neighbors(
    col: int,
    row: int,
    topology: str,
    n_rows: int,
    n_columns: int,
    labels_coords_unique: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return the topological neighbors of a neuron."""

    # Return common topological neighbors for the two grid types
    topological_neighbors = [
        (col - 1, row),
        (col + 1, row),
        (col, row - 1),
        (col, row + 1),
    ]

    # Append extra topological neighbors for hexagonal grid type
    if topology == 'hexagonal':
        offset = (-1) ** row
        topological_neighbors += [
            (col - offset, row - offset),
            (col - offset, row + offset),
        ]

    # Apply constraints
    topological_neighbors = [
        (col, row)
        for col, row in topological_neighbors
        if 0 <= col < n_columns and 0 <= row < n_rows and (col, row) in labels_coords_unique
    ]

    return topological_neighbors


class SOM(BaseEstimator, ClusterMixin):
    """Class to fit and visualize a Self-Organizing Map (SOM).

    The implementation uses MiniSom from minisom. Read more in the
    [user_guide].

    Args:
        n_columns:
            The number of columns in the map.

        n_rows:
            The number of rows in the map.

        sigma:
            Spread of the neighborhood function.

        learning_rate:
            Initial learning rate.

        decay_function:
            Function that reduces learning_rate and sigma at each iteration.

        neighborhood_function:
            Function that weights the neighborhood of a position in the map.
            Possible values: 'gaussian', 'mexican_hat', 'bubble', 'triangle'.

        topology:
            Topology of the map. Possible values: 'rectangular', 'hexagonal'.

        activation_distance:
            Distance used to activate the map. Possible values: 'euclidean',
            'cosine', 'manhattan', 'chebyshev'.

        random_state:
            Control the randomization of the algorithm.

            - If int, `random_state` is the seed used by the random number
            generator.
            - If `RandomState` instance, random_state is the random number
            generator.
            - If `None`, the random number generator is the `RandomState`
            instance used by `np.random`.
    """

    def __init__(
        self: Self,
        n_columns: int | None = None,
        n_rows: int | None = None,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        decay_function: Callable = asymptotic_decay,
        neighborhood_function: str = 'gaussian',
        topology: str = 'rectangular',
        activation_distance: str | Callable = 'euclidean',
        random_state: np.random.RandomState | int | None = None,
    ) -> None:
        self.n_columns = n_columns
        self.n_rows = n_rows
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay_function = decay_function
        self.neighborhood_function = neighborhood_function
        self.topology = topology
        self.activation_distance = activation_distance
        self.random_state = random_state

    def _generate_neighbors(
        self: Self,
        labels_coords_unique: list[tuple[int, int]],
        labels_mapping: dict[tuple[int, int], int],
    ) -> npt.NDArray:
        """Generate pairs of neighboring labels."""

        # Generate grid topological neighbors
        topological_neighbors = [
            product(
                [label_coords],
                extract_topological_neighbors(
                    *label_coords,
                    self.topology,
                    self.n_rows_,
                    self.n_columns_,
                    labels_coords_unique,
                ),
            )
            for label_coords in labels_coords_unique
        ]

        # Flatten grid topological neighbors
        topological_neighbors_flat = cast(
            list[tuple[tuple[int, int], tuple[int, int]]],
            [pair for pairs in topological_neighbors for pair in pairs],
        )

        # Generate cluster neighbors
        all_neighbors = sorted(
            {(labels_mapping[pair[0]], labels_mapping[pair[1]]) for pair in topological_neighbors_flat},
        )

        # Keep unique unordered pairs
        neighbors = []
        for pair in all_neighbors:
            if pair not in neighbors and pair[::-1] not in neighbors:
                neighbors.append(pair)

        return np.array(neighbors)

    def fit(self: Self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params: dict[str, Any]) -> Self:
        """Train the self-organizing map.

        Args:
            X:
                Training instances to cluster.

            y:
                Ignored.

            fit_params:
                Parameters to pass to train method of the MiniSom object.

                The following parameters can be used:

                num_iteration: If `use_epochs` is `False`, the weights will be
                updated `num_iteration` times. Otherwise they will be updated
                `len(X) * num_iteration` times.

                random_order:
                If `True`, samples are picked in random order.
                Otherwise the samples are picked sequentially.

                verbose:
                If `True` the status of the training will be
                printed each time the weights are updated.

                use_epochs:
                If `True` the SOM will be trained for num_iteration epochs.
                In one epoch the weights are updated `len(data)` times and
                the learning rate is constat throughout a single epoch.

        Returns:
            The object itself.
        """
        # Check random state
        self.random_state_ = check_random_state(self.random_state).randint(low=np.iinfo(np.int32).max)

        # Check and normalize input data
        X_scaled = minmax_scale(check_array(X, dtype=np.float32))

        # Initialize size
        n_neurons = 5 * np.sqrt(X_scaled.shape[0])
        if self.n_rows is None and self.n_columns is None:
            self.n_rows_ = self.n_columns_ = int(np.ceil(np.sqrt(n_neurons)))
        elif self.n_rows is None and self.n_columns is not None:
            self.n_columns_ = self.n_columns
            self.n_rows_ = int(np.ceil(n_neurons / self.n_columns_))
        elif self.n_columns is None and self.n_rows is not None:
            self.n_rows_ = self.n_rows
            self.n_columns_ = int(np.ceil(n_neurons / self.n_rows_))
        elif self.n_columns is not None and self.n_rows is not None:
            self.n_rows_ = self.n_rows
            self.n_columns_ = self.n_columns

        # Create MiniSom object
        self.algorithm_ = MiniSom(
            x=self.n_rows_,
            y=self.n_columns_,
            input_len=X_scaled.shape[1],
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            decay_function=self.decay_function,
            neighborhood_function=self.neighborhood_function,
            topology=self.topology,
            activation_distance=self.activation_distance,
            random_seed=self.random_state_,
        )

        # Fit MiniSom
        if 'num_iteration' not in fit_params:
            fit_params = {**fit_params, 'num_iteration': cast(Any, 1000)}
        self.algorithm_.train(data=X_scaled, **fit_params)

        # Grid labels
        labels_coords = [(int(i), int(j)) for i, j in [self.algorithm_.winner(x_scaled) for x_scaled in X_scaled]]

        # Generate labels mapping
        self.labels_mapping_ = generate_labels_mapping(labels_coords)

        # Generate cluster labels
        self.labels_ = np.array(
            [self.labels_mapping_[grid_label] for grid_label in labels_coords],
        )

        # Generate labels neighbors
        self.neighbors_ = self._generate_neighbors(
            sorted(set(labels_coords)),
            self.labels_mapping_,
        )

        return self

    def fit_predict(
        self: Self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: dict[str, Any],
    ) -> npt.NDArray:
        """Train the self-organizing map and assign cluster labels to samples.

        Args:
            X:
                New data to transform.

            y:
                Ignored.

            fit_params:
                Parameters to pass to train method of the MiniSom object.

                The following parameters can be used:

                num_iteration: If `use_epochs` is `False`, the weights will be
                updated `num_iteration` times. Otherwise they will be updated
                `len(X) * num_iteration` times.

                random_order:
                If `True`, samples are picked in random order.
                Otherwise the samples are picked sequentially.

                verbose:
                If `True` the status of the training will be
                printed each time the weights are updated.

                use_epochs:
                If `True` the SOM will be trained for num_iteration epochs.
                In one epoch the weights are updated `len(data)` times and
                the learning rate is constat throughout a single epoch.

        Returns:
            labels:
                Index of the cluster each sample belongs to.
        """
        return self.fit(X=X, y=None, **fit_params).labels_
