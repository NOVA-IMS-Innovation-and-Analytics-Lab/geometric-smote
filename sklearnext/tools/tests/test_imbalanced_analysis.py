"""
Test the imbalanced_analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

from ...model_selection import ModelSearchCV
from ...utils.validation import check_oversamplers_classifiers
from ...tools.imbalanced_analysis import (
    summarize_binary_datasets,
    _define_binary_experiment_parameters,
    _calculate_results
)

X1, y1 = make_classification(weights=[0.90, 0.10], n_samples=100, random_state=0)
X2, y2 = make_classification(weights=[0.80, 0.20], n_samples=100, n_features=10, random_state=1)
DATASETS = [('A',(X1, y1)), ('B', (X2, y2))]
OVERSAMPLERS = [
    ('random', RandomOverSampler()),
    ('smote', SMOTE(), {'k_neighbors': [2, 3]}),
    ('adasyn', ADASYN(), {'n_neighbors': [2, 3, 4]})
]
CLASSIFIERS = [
    ('knn', KNeighborsClassifier()),
    ('dtc', DecisionTreeClassifier(), {'max_depth': [3, 5]})
]
N_RUNS = 3
ESTIMATORS, PARAM_GRIDS = check_oversamplers_classifiers(OVERSAMPLERS, CLASSIFIERS, n_runs=N_RUNS, random_state=0).values()
MSCV = ModelSearchCV(ESTIMATORS, PARAM_GRIDS, scoring=None, iid=True, refit=False,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0), error_score='raise',
                    return_train_score=False, scheduler=None, n_jobs=-1, cache_cv=True, verbose=False)


def test_summarize_binary_datasets():
    """Test the output of summarize binary
    datasets function."""
    datasets_summary = summarize_binary_datasets(DATASETS)
    expected_datasets_summary = pd.DataFrame(
        {
            'Dataset name': ['B', 'A'],
            'Features': [10, 20],
            'Instances': [100, 100],
            'Minority instances': [20, 10],
            'Majority instances': [80, 90],
            'Imbalance Ratio': [4.0, 9.0]
        }
    )
    pd.testing.assert_frame_equal(datasets_summary, expected_datasets_summary, check_dtype=False)


@pytest.mark.parametrize('scoring,expected_parameters', [
    (None, (None, ['mean_test_score'], 'classifier')),
    ('accuracy', ('accuracy', ['mean_test_score'], 'classifier')),
    (['accuracy', 'f1'], (['accuracy', 'f1'], ['mean_test_accuracy', 'mean_test_f1'], 'classifier'))
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
