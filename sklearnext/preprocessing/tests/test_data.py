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
    selector = FeatureSelector(indices=indices)
    X_t = selector.fit_transform(X, y)
    assert X_t.shape[0] == X.shape[0]
    assert X_t.shape[1] == len(indices)
    assert np.array_equal(X_t, X[:, indices])


def test_default_feature_selector():
    selector = FeatureSelector()
    X_t = selector.fit_transform(X, y)
    assert np.array_equal(X_t, X)


def test_feature_selector_pipeline_integration():
    pipeline = Pipeline([('selector', FeatureSelector(indices=[0, 2])), ('lr', LinearRegression())])
    pipeline.fit(X, y)


def test_feature_selector_set_parameters():
    indices, updated_indices = [0,3,4], None

    selector = FeatureSelector(indices)
    X_t = selector.fit_transform(X, y)
    assert X_t.shape[1] == len(indices)

    selector.set_params(indices=updated_indices)
    X_t = selector.fit_transform(X, y)
    assert np.array_equal(X_t, X)


@pytest.mark.parametrize('ratio,random_state', [
    (0.9, None),
    (0.4, 5),
    (0.1, np.random.RandomState(2))
])
def test_row_selector_random(ratio, random_state):
    selector = RowSelector(ratio, random_state)
    X_t, y_t = selector.fit_sample(X, y)
    n_samples = int(ratio * len(X))
    assert X_t.shape == (n_samples, X.shape[1])
    assert y_t.shape == (n_samples,)
    # Assert corresondance X, y

@pytest.mark.parametrize('ratio', [
    0.9, 0.4, 0.1, 1.0
])
def test_row_selector_head(ratio):
    selector = RowSelector(ratio, random_state='head')
    X_t, y_t = selector.fit_sample(X, y)
    n_samples = int(ratio * len(X))
    assert np.array_equal(X_t, X[:n_samples])
    assert np.array_equal(y_t, y[:n_samples])


@pytest.mark.parametrize('ratio', [
    0.2, 0.7, 0.5, 1.0
])
def test_row_selector_tail(ratio):
    selector = RowSelector(ratio, random_state='tail')
    X_t, y_t = selector.fit_sample(X, y)
    n_samples = int(ratio * len(X))
    assert np.array_equal(X_t, X[-n_samples:])
    assert np.array_equal(y_t, y[-n_samples:])


def test_default_row_selector():
    selector = RowSelector()
    X_t, y_t = selector.fit_sample(X, y)
    assert np.array_equal(X_t, X)
    assert np.array_equal(y_t, y)


@pytest.mark.parametrize('indices', [
    np.arange(0, 10), np.arange(10, 100), np.arange(0, len(X))[::-1]
])
def test_row_selector_different_input_data(indices):
    selector = RowSelector(ratio=0.5)
    selector.fit(X, y)
    with pytest.raises(RuntimeError):
        selector.sample(X[indices], y[indices])


def test_row_selector_pipeline_integration():
    pipeline = Pipeline(
        [
            ('selector', RowSelector(ratio=0.8, random_state=0)),
            ('lr', LinearRegression())
        ]
    )
    pipeline.fit(X, y)