"""
Test the model_analysis module.
"""

import pytest
from numpy.testing import assert_array_equal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from ..model_analysis import report_model_search_results
from ...model_selection import ModelSearchCV

X_clf, y_clf = make_classification()
CLASSIFIERS = [
    ('logr', LogisticRegression()),
    ('svc', SVC()),
    ('clf_pip', Pipeline([('scaler', MinMaxScaler()), ('lr', LogisticRegression())]))
]
CLASSIFIERS_PARAM_GRIDS = {'svc__C': [0.01, 0.1, 1.0], 'svc__kernel': ['rbf', 'linear']}
BASIC_COLUMNS = ['models', 'params', 'mean_fit_time']


@pytest.mark.parametrize('scoring,sort_results', [
    (None, None),
    (None, 'mean_fit_time'),
    (None, 'mean_test_score'),
    ('accuracy', None),
    ('f1', 'mean_fit_time'),
    ('f1', 'mean_test_score'),
    (['accuracy', 'f1'], None),
    (['accuracy', 'f1'], 'mean_fit_time'),
    (['accuracy', 'f1'], 'mean_test_accuracy'),
    (['accuracy', 'f1'], 'mean_test_f1')
])
def test_report_model_search_results(scoring, sort_results):
    """Test the output of the model search report function."""
    mscv = ModelSearchCV(CLASSIFIERS, CLASSIFIERS_PARAM_GRIDS, scoring=scoring, refit=False)
    mscv.fit(X_clf, y_clf)
    report = report_model_search_results(mscv, sort_results)
    assert len(report.columns) == (len(mscv.scorer_) if isinstance(mscv.scoring, list) else 1) + len(BASIC_COLUMNS)
    if sort_results:
        assert_array_equal(report[sort_results], report[sort_results].sort_values(ascending=(sort_results == 'mean_fit_time')))