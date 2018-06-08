"""
Test the data module.
"""

import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from ..validation import _normalize_param_grid, check_param_grids, check_datasets

X, y = make_regression()
ESTIMATORS = [
    ('lr', LinearRegression()),
    ('svr', SVR()),
    ('pip', Pipeline([('scaler', MinMaxScaler()), ('lr', LinearRegression())]))
]
PARAM_GRIDS = [
    {'lr__normalize': [True, False], 'lr__fit_intercept': [True, False]},
    {'svr__C': [0.01, 0.1, 1.0], 'svr__kernel': ['rbf', 'linear']},
    {'pip__scaler__feature_range': [(0, 1), (0, 10)], 'pip__lr__normalize': [True, False]}
]
UPDATED_PARAM_GRIDS = [
    {'lr__normalize': [True, False], 'lr__fit_intercept': [True, False], 'est_name':['lr']},
    {'svr__C': [0.01, 0.1, 1.0], 'svr__kernel': ['rbf', 'linear'], 'est_name':['svr']},
    {'pip__scaler__feature_range': [(0, 1), (0, 10)], 'pip__lr__normalize': [True, False], 'est_name':['pip']}
]


@pytest.mark.parametrize('param_grid,updated_param_grid', [
    (PARAM_GRIDS[0], UPDATED_PARAM_GRIDS[0]),
    (PARAM_GRIDS[1], UPDATED_PARAM_GRIDS[1]),
    (PARAM_GRIDS[2], UPDATED_PARAM_GRIDS[2]),
    ({}, {'est_name': []})
])
def test_check_normalize_grid(param_grid, updated_param_grid):
    assert _normalize_param_grid(param_grid) == updated_param_grid


@pytest.mark.parametrize('param_grids,updated_param_grids,estimators', [
    (PARAM_GRIDS, UPDATED_PARAM_GRIDS, ESTIMATORS),
    (PARAM_GRIDS[0:2], UPDATED_PARAM_GRIDS[0:2] + [{'est_name': ['pip']}], ESTIMATORS),
    (PARAM_GRIDS[0:1], UPDATED_PARAM_GRIDS[0:1] + [{'est_name': ['svr']}, {'est_name': ['pip']}], ESTIMATORS),
    (PARAM_GRIDS[0:1] + PARAM_GRIDS[2:], UPDATED_PARAM_GRIDS[0:1] + UPDATED_PARAM_GRIDS[2:] + [{'est_name': ['svr']}], ESTIMATORS),
    ({}, [{'est_name': ['lr']}, {'est_name': ['svr']}, {'est_name': ['pip']}], ESTIMATORS)
])
def test_check_param_grids(param_grids, updated_param_grids, estimators):
    checked_param_grids = check_param_grids(param_grids, estimators)
    assert len(checked_param_grids) == len(updated_param_grids)
    assert all(param_grid in checked_param_grids for param_grid in updated_param_grids)


@pytest.mark.parametrize('dataset_name1,dataset_name2,data', [
    ('ds1', 'ds1', (X, y)),
    ('ds1', X, (X, y)),
])
def test_check_datasets(dataset_name1, dataset_name2, data):
    with pytest.raises(ValueError):
        check_datasets([(dataset_name1, data), (dataset_name2, data)])