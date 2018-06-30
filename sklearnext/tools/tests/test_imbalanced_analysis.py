"""
Test the imbalanced_analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearnext.model_selection import ModelSearchCV
from sklearnext.utils.validation import check_oversamplers_classifiers
from sklearnext.tools.imbalanced_analysis import (
    summarize_binary_datasets,
    _define_binary_experiment_parameters,
    _calculate_results
)

X1, y1 = make_classification(weights=[0.90, 0.10], n_samples=100, random_state=0)
X2, y2 = make_classification(weights=[0.80, 0.20], n_samples=100, n_features=10, random_state=1)
DATASETS = [('A',(X1, y1)), ('B', (X2, y2))]
OVERSAMPLERS = [
    ('random', RandomOverSampler()),
    ('smote', SMOTE(), {'k_neighbors': [2, 3], 'kind': ['regular', 'borderline1']}),
    ('adasyn', ADASYN(), {'n_neighbors': [2, 3, 4]})
]
CLASSIFIERS = [
    ('lr', LogisticRegression()),
    ('svc', SVC(), {'C': [0.1, 1.0]})
]
GROUP_KEYS = ['Dataset', 'Oversampler', 'Classifier', 'params']
N_RUNS = 3
ESTIMATORS, PARAM_GRIDS = check_oversamplers_classifiers(OVERSAMPLERS, CLASSIFIERS, n_runs=N_RUNS, random_state=0).values()
MSCV = ModelSearchCV(ESTIMATORS, PARAM_GRIDS, scoring=None, iid=True, refit=False,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), error_score='raise',
                    return_train_score=False, scheduler=None, n_jobs=-1, cache_cv=True, verbose=False)


def test_summarize_binary_datasets():
    """Test the output of summarize binary
    datasets function."""
    np.testing.assert_array_equal(summarize_binary_datasets(DATASETS),
                                  pd.DataFrame({
                                      'Dataset name': ['A', 'B'],
                                      '# features': [20, 10],
                                      '# instances': [100, 100],
                                      '# minority instances': [10, 20],
                                      '# majority instances': [90, 80],
                                      'Imbalance Ratio': [9.0, 4.0]
                                  }))


@pytest.mark.parametrize('scoring,expected_parameters', [
    (None, (None, ['mean_test_score'], GROUP_KEYS, 'classifier')),
    ('accuracy', ('accuracy', ['mean_test_score'], GROUP_KEYS, 'classifier')),
    (['accuracy', 'f1'], (['accuracy', 'f1'], ['mean_test_accuracy', 'mean_test_f1'], GROUP_KEYS, 'classifier'))
])
def test_define_binary_experiment_parameters(scoring, expected_parameters):
    """Test the definition of the binary experimet parameters."""
    MSCV.set_params(scoring=scoring)
    assert expected_parameters == _define_binary_experiment_parameters(MSCV)


@pytest.mark.parametrize('scoring,expected_results_columns', [
    (None, ['models', 'params', 'mean_test_score', 'Dataset']),
    ('accuracy', ['models', 'params', 'mean_test_score', 'Dataset']),
    (['accuracy', 'recall'], ['models', 'params', 'mean_test_accuracy', 'mean_test_recall', 'Dataset'])
])
def test_calculate_results(scoring, expected_results_columns):
    """Test the calculation of the experiment's results."""
    MSCV.set_params(scoring=scoring)
    scoring_cols = _define_binary_experiment_parameters(MSCV)[1]
    results = _calculate_results(MSCV, DATASETS, scoring_cols, verbose=False)
    assert len(results) == len(ParameterGrid(MSCV.param_grids)) * len(DATASETS)
    assert expected_results_columns == results.columns.get_level_values(0).tolist()
