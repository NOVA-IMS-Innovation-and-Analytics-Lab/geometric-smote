"""
Test the imbalanced_analysis module.
"""

from os import remove
from pickle import load

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.pipeline import Pipeline

from ...model_selection import ModelSearchCV
from ...utils.validation import check_oversamplers_classifiers
from ...tools.imbalanced_analysis import BinaryExperiment, GROUP_KEYS

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
EXPERIMENT = BinaryExperiment('test_experiment', OVERSAMPLERS, CLASSIFIERS, DATASETS, 
                              scoring=None, n_splits=3, n_runs=3, random_state=0)


@pytest.mark.parametrize('scoring,n_runs', [
    (None, 2),
    ('accuracy', 3),
    (['accuracy', 'recall'], 4)
])
def test_experiment_initialization(scoring, n_runs):
    """Test the initialization of experiment's parameters."""
    experiment = BinaryExperiment('test_experiment', OVERSAMPLERS, CLASSIFIERS, DATASETS, 
                                  scoring=scoring, n_splits=3, n_runs=n_runs, random_state=0)
    experiment._initialize(1, 0)
    if not isinstance(scoring, list):
        assert experiment.scoring_cols_ == ['mean_test_score']
    else:
        assert experiment.scoring_cols_ == ['mean_test_%s' % scorer for scorer in scoring]
    assert experiment.datasets_names_ == ('A', 'B')
    assert experiment.oversamplers_names_ == ('random', 'smote', 'adasyn')
    assert experiment.classifiers_names_ == ('knn', 'dtc')
    assert len(experiment.estimators_) == len(experiment.param_grids_) == len(OVERSAMPLERS) * len(CLASSIFIERS) * n_runs
    

def test_datasets_summary():
    """Test the dataset's summary."""
    EXPERIMENT = BinaryExperiment('test_experiment', OVERSAMPLERS, CLASSIFIERS, DATASETS, 
                                  scoring=None, n_splits=3, n_runs=3, random_state=0)
    EXPERIMENT.summarize_datasets()
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
    pd.testing.assert_frame_equal(EXPERIMENT.datasets_summary_, expected_datasets_summary, check_dtype=False)


def test_experiment_raised_errors():
    """Test if an attribute error is raised."""
    for func in ('calculate_optimal_results', 'calculate_wide_optimal_results', 'calculate_ranking_results', 
                 'calculate_mean_ranking_results', 'calculate_friedman_test_results', 'calculate_adjusted_pvalues_results'):
        with pytest.raises(AttributeError):
            getattr(EXPERIMENT, func)()

def test_results():
    """Test the results of experiment."""
    EXPERIMENT.run()
    assert EXPERIMENT.results_.reset_index().columns.get_level_values(0).tolist()[:-2] == GROUP_KEYS
    assert len(EXPERIMENT.results_) == len(DATASETS) * len(ParameterGrid(EXPERIMENT.param_grids_)) // EXPERIMENT.n_runs


def test_optimal_results():
    """Test the optimal results of experiment."""
    EXPERIMENT.calculate_optimal_results()
    assert set(EXPERIMENT.optimal_results_.Dataset.unique()) == set(EXPERIMENT.datasets_names_)
    assert set(EXPERIMENT.optimal_results_.Oversampler.unique()) == set(EXPERIMENT.oversamplers_names_)
    assert set(EXPERIMENT.optimal_results_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert len(EXPERIMENT.optimal_results_) == len(DATASETS) * len(OVERSAMPLERS) * len(CLASSIFIERS)


def test_wide_optimal_results():
    """Test the wide optimal results of experiment."""
    EXPERIMENT.calculate_wide_optimal_results()
    assert set(EXPERIMENT.wide_optimal_results_.Dataset.unique()) == set(EXPERIMENT.datasets_names_)
    assert set(EXPERIMENT.wide_optimal_results_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.wide_optimal_results_.columns)
    assert len(EXPERIMENT.wide_optimal_results_) == len(DATASETS) * len(CLASSIFIERS)


def test_ranking_results():
    """Test the ranking results of experiment."""
    EXPERIMENT.calculate_ranking_results()
    assert set(EXPERIMENT.ranking_results_.Dataset.unique()) == set(EXPERIMENT.datasets_names_)
    assert set(EXPERIMENT.ranking_results_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.ranking_results_.columns)
    assert len(EXPERIMENT.ranking_results_) == len(DATASETS) * len(CLASSIFIERS)

def test_mean_ranking_results():
    """Test the mean_ranking results of experiment."""
    EXPERIMENT.calculate_mean_ranking_results()
    assert set(EXPERIMENT.mean_ranking_results_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.mean_ranking_results_.columns)
    assert len(EXPERIMENT.mean_ranking_results_) == len(CLASSIFIERS)


def test_friedman_test_results():
    """Test the results of friedman test."""
    EXPERIMENT.calculate_friedman_test_results()
    assert set(EXPERIMENT.friedman_test_results_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert len(EXPERIMENT.friedman_test_results_) == len(CLASSIFIERS)


def test_dump():
    """Test the dump method."""
    EXPERIMENT.dump()
    file_name = f'{EXPERIMENT.name}.pkl'
    with open(file_name, 'rb') as file:
        experiment = load(file)
    for attr in ('results_', 'optimal_results_', 'wide_optimal_results_', 'ranking_results_', 'mean_ranking_results_', 'friedman_test_results_'):
        pd.testing.assert_frame_equal(getattr(EXPERIMENT, attr), getattr(experiment, attr))
    remove(file_name)
