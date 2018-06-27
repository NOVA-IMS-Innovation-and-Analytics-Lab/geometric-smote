"""
Test the data module.
"""

import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR, SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from ...utils.validation import (
    _normalize_param_grid,
    check_param_grids,
    check_datasets,
    check_random_states,
    check_oversamplers_classifiers
)

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
OVERSAMPLERS = [
    ('random', RandomOverSampler()),
    ('smote', SMOTE(), {'k_neighbors': [2, 3, 4], 'kind': ['regular', 'borderline1']})
]
CLASSIFIERS = [
    ('svc', SVC(), {'C': [0.1, 0.5, 1.0], 'kernel': ['rbf', 'linear']})
]
OVERSAMPLERS_CLASSIFIERS = [
    ('random|svc_0', Pipeline([('random', RandomOverSampler()), ('svc', SVC())])),
    ('random|svc_1', Pipeline([('random', RandomOverSampler()), ('svc', SVC())])),
    ('smote|svc_0', Pipeline([('smote', RandomOverSampler()), ('svc', SVC())])),
    ('smote|svc_1', Pipeline([('smote', RandomOverSampler()), ('svc', SVC())]))
]
OVERSAMPLERS_CLASSIFIERS_PARAM_GRIDS = [
    {
        'est_name': ['random|svc_0'],
        'random_state': None,
        'random|svc_0__svc__C': [0.1, 0.5, 1.0],
        'random|svc_0__svc__kernel': ['rbf', 'linear']
    },
    {
        'est_name': ['random|svc_1'],
        'random_state': None,
        'random|svc_1__svc__C': [0.1, 0.5, 1.0],
        'random|svc_1__svc__kernel': ['rbf', 'linear']
    },
    {
        'est_name': ['smote|svc_0'],
        'random_state': None,
        'smote|svc_0__svc__C': [0.1, 0.5, 1.0],
        'smote|svc_0__svc__kernel': ['rbf', 'linear'],
        'smote|svc_0__svc__C': [0.1, 0.5, 1.0],
        'smote|svc_0__smote__k_neighbors': [2, 3, 4],
        'smote|svc_0__smote__kind': ['regular', 'borderline1']
    },
    {
        'est_name': ['smote|svc_1'],
        'random_state': None,
        'smote|svc_1__svc__C': [0.1, 0.5, 1.0],
        'smote|svc_1__svc__kernel': ['rbf', 'linear'],
        'smote|svc_1__svc__C': [0.1, 0.5, 1.0],
        'smote|svc_1__smote__k_neighbors': [2, 3, 4],
        'smote|svc_1__smote__kind': ['regular', 'borderline1']
    }

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


@pytest.mark.parametrize('oversamplers,classifiers,n_runs,random_state,estimators,param_grids', [
    (OVERSAMPLERS, CLASSIFIERS, 2, 1, OVERSAMPLERS_CLASSIFIERS, OVERSAMPLERS_CLASSIFIERS_PARAM_GRIDS)
])
def test_check_oversamplers_classifiers(oversamplers, classifiers, n_runs, random_state, estimators, param_grids):

    generated_estimators, generated_param_grids = check_oversamplers_classifiers(
        oversamplers, classifiers, n_runs, random_state).values()

    generated_est_names, generated_estimators = zip(*generated_estimators)
    generated_params = [est.get_params(False).keys() for est in generated_estimators]
    est_names, estimators = zip(*estimators)
    params = [est.get_params(False).keys() for est in estimators]

    random_states = check_random_states(random_state, len(oversamplers) * len(classifiers) * n_runs)
    final_param_grids = []
    for param_grid, random_state in zip(param_grids, random_states):
        param_grid = param_grid.copy()
        param_grid.update({'random_state': random_state})
        final_param_grids.append(param_grid)

    assert generated_est_names == est_names
    assert [set(list(pars)) for pars in generated_params] == [set(list(pars)) for pars in params]
    assert generated_param_grids == final_param_grids


@pytest.mark.parametrize('dataset_name1,dataset_name2,data', [
    ('ds1', 'ds1', (X, y)),
    ('ds1', X, (X, y)),
])
def test_check_datasets(dataset_name1, dataset_name2, data):
    with pytest.raises(ValueError):
        check_datasets([(dataset_name1, data), (dataset_name2, data)])