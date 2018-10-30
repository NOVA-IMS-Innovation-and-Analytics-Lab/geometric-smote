"""
Test the data module.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from imblearn.pipeline import Pipeline

from ...preprocessing import FeatureSelector, RowSelector

X, y = make_regression(n_features=10)


@pytest.mark.parametrize('indices', [
    ([0,1,2]),
    ([5]),
    (np.arange(0, 10))
])
def test_feature_selector(indices):
    """Test the feature selector."""
    selector = FeatureSelector(indices=indices)
    X_t = selector.fit_transform(X, y)
    assert X_t.shape[0] == X.shape[0]
    assert X_t.shape[1] == len(indices)
    assert np.array_equal(X_t, X[:, indices])


def test_default_feature_selector():
    """Test the default feature selector."""
    selector = FeatureSelector()
    X_t = selector.fit_transform(X, y)
    assert np.array_equal(X_t, X)


def test_feature_selector_pipeline_integration():
    """Test the integration of feature selector
    and pipelines."""
    pipeline = Pipeline([('selector', FeatureSelector(indices=[0, 2])), ('lr', LinearRegression())])
    pipeline.fit(X, y)


def test_feature_selector_set_parameters():
    """Test the feature selector set of
    parameters method."""
    indices, updated_indices = [0, 3, 4], None

    selector = FeatureSelector(indices)
    X_t = selector.fit_transform(X, y)
    assert X_t.shape[1] == len(indices)

    selector.set_params(indices=updated_indices)
    X_t = selector.fit_transform(X, y)
    assert np.array_equal(X_t, X)


@pytest.mark.parametrize('sampling_strategy,selection_strategy', [
    (0.9, None),
    (0.4, 5),
    (0.1, np.random.RandomState(2))
])
def test_row_selector_strategy(sampling_strategy, selection_strategy):
    """Test the selection strategy of row selector."""
    selector = RowSelector(sampling_strategy, selection_strategy)
    X_t, y_t = selector.fit_resample(X, y)
    n_samples = int(sampling_strategy * len(X))
    assert X_t.shape == (n_samples, X.shape[1])
    assert y_t.shape == (n_samples,)


@pytest.mark.parametrize('sampling_strategy', [
    0.9, 0.4, 0.1, 1.0
])
def test_row_selector_head(sampling_strategy):
    """Test the head strategy of row selector."""
    selector = RowSelector(sampling_strategy, selection_strategy='head')
    X_t, y_t = selector.fit_resample(X, y)
    n_samples = int(sampling_strategy * len(X))
    assert np.array_equal(X_t, X[:n_samples])
    assert np.array_equal(y_t, y[:n_samples])


@pytest.mark.parametrize('sampling_strategy', [
    0.2, 0.7, 0.5, 1.0
])
def test_row_selector_tail(sampling_strategy):
    """Test the tail strategy of row selector."""
    selector = RowSelector(sampling_strategy, selection_strategy='tail')
    X_t, y_t = selector.fit_resample(X, y)
    n_samples = int(sampling_strategy * len(X))
    assert np.array_equal(X_t, X[-n_samples:])
    assert np.array_equal(y_t, y[-n_samples:])


def test_default_row_selector():
    """Test the default row selector."""
    selector = RowSelector()
    X_t, y_t = selector.fit_resample(X, y)
    assert np.array_equal(X_t, X)
    assert np.array_equal(y_t, y)


def test_row_selector_pipeline_integration():
    """Test the integration of row selector
    and pipelines."""
    pipeline = Pipeline(
        [
            ('selector', RowSelector(sampling_strategy=0.8, selection_strategy=0)),
            ('lr', LinearRegression())
        ]
    )
    pipeline.fit(X, y)
