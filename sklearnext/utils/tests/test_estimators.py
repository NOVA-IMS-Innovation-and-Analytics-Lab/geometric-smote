"""
Test the estimators module.
"""

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression, make_classification

from ..estimators import _ParametrizedEstimatorsMixin
from ...utils import _ParametrizedClassifiers, _ParametrizedRegressors

X_reg, y_reg = make_regression()
X_clf, y_clf = make_classification()
REGRESSORS = [
    ('linr', LinearRegression()),
    ('svr', SVR()),
    ('reg_pip', Pipeline([('scaler', MinMaxScaler()), ('lr', LinearRegression())]))
]
CLASSIFIERS = [
    ('logr', LogisticRegression()),
    ('svc', SVC()),
    ('clf_pip', Pipeline([('scaler', MinMaxScaler()), ('lr', LogisticRegression())]))
]
REGRESSORS_PARAM_GRIDS = [
    {'linr__normalize': [True, False], 'linr__fit_intercept': [True, False]},
    {'svr__C': [0.01, 0.1, 1.0], 'svr__kernel': ['rbf', 'linear']},
    {'reg_pip__scaler__feature_range': [(0, 1), (0, 10)], 'reg_pip__lr__normalize': [True, False]}
]
CLASSIFIERS_PARAM_GRIDS = {'svc__C': [0.01, 0.1, 1.0], 'svc__kernel': ['rbf', 'linear']}


def _generate_expected_params(estimators):
    expected_params = {'est_name': None, 'random_state': None, 'estimators': estimators}
    for est_name, step in estimators:
        expected_params[est_name] = step
        est_params = step.get_params()
        for param, value in est_params.items():
            expected_params['{}__{}'.format(est_name, param)] = value
    return expected_params


@pytest.mark.parametrize('estimators', [
    (None),
    ([]),
    ([('est', LinearRegression()), ('est', SVR())])
])
def test_parametrized_estimators_initialization(estimators):
    """Test the initialization of parametrized estimators class."""
    with pytest.raises(Exception):
        _ParametrizedEstimatorsMixin(estimators)


@pytest.mark.parametrize('estimators,updated_params', [
    (REGRESSORS, {'linr': LinearRegression(fit_intercept=False)}),
    (REGRESSORS, {'svr': SVR(C=2.0)}),
])
def test_parametrized_estimators_params_methods(estimators, updated_params):
    """Test the set and get parameters methods."""
    pe = _ParametrizedEstimatorsMixin(estimators)
    pe.set_params(**updated_params)
    assert pe.get_params() == _generate_expected_params(pe.estimators)


@pytest.mark.parametrize('estimators,est_name,X,y', [
    (REGRESSORS, 'linr', X_reg, y_reg),
    (REGRESSORS, 'svr', X_reg, y_reg),
    (REGRESSORS, 'reg_pip', X_reg, y_reg),
    (CLASSIFIERS, 'svc', X_clf, y_clf),
])
def test_parametrized_estimators_fitting(estimators, est_name, X, y):
    """Test parametrized estimators fitting process."""
    pe = _ParametrizedEstimatorsMixin(estimators, est_name)
    pe.fit(X, y)
    fitted_estimator = dict(estimators).get(est_name)
    assert isinstance(fitted_estimator, pe.estimator_.__class__)


@pytest.mark.parametrize('estimators,X,y,est_name', [
    (REGRESSORS, X_reg, y_reg, None),
    (CLASSIFIERS, X_clf, y_clf, None),
])
def test_parametrized_estimators_fitting_error(estimators, X, y, est_name):
    """Test parametrized estimators fitting error."""
    with pytest.raises(ValueError):
        _ParametrizedEstimatorsMixin(estimators, est_name).fit(X, y)


def test_parametrized_classifiers_type():
    """Test parametrized classifiers type of estimator."""
    pf = _ParametrizedClassifiers(CLASSIFIERS)
    assert pf._estimator_type == 'classifier'


def test_parametrized_regressors_type():
    """Test parametrized regressors type of estimator."""
    pr = _ParametrizedRegressors(REGRESSORS)
    assert pr._estimator_type == 'regressor'