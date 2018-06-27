"""
Test the search module.
"""

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression, make_classification
from ...model_selection import ModelSearchCV


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


@pytest.mark.parametrize('estimators,param_grids,estimator_type', [
    (REGRESSORS, REGRESSORS_PARAM_GRIDS, 'regressor'),
    (CLASSIFIERS, CLASSIFIERS_PARAM_GRIDS, 'classifier'),
])
def test_model_search_cv_type(estimators, param_grids, estimator_type):
    """Test model search cv estimator type."""
    mscv = ModelSearchCV(estimators, param_grids)
    assert mscv._estimator_type == estimator_type