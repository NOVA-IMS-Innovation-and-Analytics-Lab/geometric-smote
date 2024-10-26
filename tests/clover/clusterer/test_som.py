"""Test the _som module."""

import numpy as np
from imblearn_extra.clover.clusterer import SOM, extract_topological_neighbors, generate_labels_mapping
from sklearn.datasets import make_classification

RANDOM_STATE = 5
X, _ = make_classification(random_state=RANDOM_STATE, n_samples=1000)


def test_generate_labels_mapping():
    """Test the generation of the labels mapping."""
    grid_labels = [(1, 1), (0, 0), (0, 1), (1, 0), (1, 1), (1, 0), (0, 1)]
    labels_mapping = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    assert generate_labels_mapping(grid_labels) == labels_mapping


def test_return_topological_neighbors_rectangular():
    """Test the topological neighbors of a neuron for rectangular grid type."""
    som = SOM(random_state=RANDOM_STATE).fit(X)
    labels_coords_unique = list({(int(i), int(j)) for i, j in [som.algorithm_.winner(x) for x in X]})
    assert extract_topological_neighbors(0, 0, som.topology, som.n_rows_, som.n_columns_, labels_coords_unique) == [
        (1, 0),
        (0, 1),
    ]
    assert extract_topological_neighbors(1, 1, som.topology, som.n_rows_, som.n_columns_, labels_coords_unique) == [
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
    ]


def test_return_topological_neighbors_hexagonal():
    """Test the topological neighbors of a neuron for hexagonal grid type."""
    som = SOM(random_state=RANDOM_STATE, topology='hexagonal').fit(X)
    labels_coords_unique = list({(int(i), int(j)) for i, j in [som.algorithm_.winner(x) for x in X]})
    assert extract_topological_neighbors(0, 0, som.topology, som.n_rows_, som.n_columns_, labels_coords_unique) == [
        (1, 0),
        (0, 1),
    ]
    assert extract_topological_neighbors(1, 1, som.topology, som.n_rows_, som.n_columns_, labels_coords_unique) == [
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
        (2, 2),
        (2, 0),
    ]


def test_no_fit():
    """Test the SOM initialization."""
    som = SOM(random_state=RANDOM_STATE)
    assert not hasattr(som, 'labels_')
    assert not hasattr(som, 'neighbors_')
    assert not hasattr(som, 'algorithm_')
    assert not hasattr(som, 'n_columns_')
    assert not hasattr(som, 'n_rows_')
    assert not hasattr(som, 'labels_mapping_')


def test_fit():
    """Test the SOM fitting process."""
    n_rows = 5
    n_columns = 3
    som = SOM(n_rows=n_rows, n_columns=n_columns, random_state=RANDOM_STATE)
    som.fit(X)
    assert np.array_equal(np.unique(som.labels_), np.arange(0, n_rows * n_columns))
    assert som.n_rows_ == n_rows
    assert som.n_columns_ == n_columns
    assert hasattr(som, 'neighbors_')
    assert hasattr(som, 'algorithm_')
    assert hasattr(som, 'labels_mapping_')
