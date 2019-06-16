"""
Test the imbalanced_analysis module.
"""

from os import remove
from pickle import load

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.pipeline import Pipeline

from ...model_selection import ModelSearchCV
from ...utils.validation import check_oversamplers_classifiers
from ...tools.imbalanced_analysis import BinaryExperiment, GROUP_KEYS, ATTR_NAMES

X1, y1 = make_classification(weights=[0.90, 0.10], n_samples=100, random_state=0)
X2, y2 = make_classification(weights=[0.80, 0.20], n_samples=100, n_features=10, random_state=1)
X3, y3 = make_classification(weights=[0.60, 0.40], n_samples=100, n_features=5, random_state=2)
DATASETS = [('A',(X1, y1)), ('B', (X2, y2)), ('C', (X3, y3))]
OVERSAMPLERS = [
    ('random', RandomOverSampler()),
    ('smote', SMOTE(), {'k_neighbors': [2, 3]}),
    ('adasyn', ADASYN(), {'n_neighbors': [2, 3, 4]})
]
CLASSIFIERS = [
    ('knn', KNeighborsClassifier()),
    ('dtc', DecisionTreeClassifier(), {'max_depth': [3, 5]})
]
EXPERIMENT = BinaryExperiment('test_experiment', DATASETS, OVERSAMPLERS, CLASSIFIERS, 
                              scoring=None, n_splits=3, n_runs=3, random_state=0)


@pytest.mark.parametrize('scoring,n_runs', [
    (None, 2),
    ('accuracy', 3),
    (['accuracy', 'recall'], 4)
])
def test_experiment_initialization(scoring, n_runs):
    """Test the initialization of experiment's parameters."""
    experiment = BinaryExperiment('test_experiment', DATASETS, OVERSAMPLERS, CLASSIFIERS, 
                                  scoring=scoring, n_splits=3, n_runs=n_runs, random_state=0)
    experiment._initialize(1, 0)
    if not isinstance(scoring, list):
        assert experiment.scoring_cols_ == ['mean_test_score']
    else:
        assert experiment.scoring_cols_ == ['mean_test_%s' % scorer for scorer in scoring]
    assert experiment.datasets_names_ == ('A', 'B', 'C')
    assert experiment.oversamplers_names_ == ('random', 'smote', 'adasyn')
    assert experiment.classifiers_names_ == ('knn', 'dtc')
    assert len(experiment.estimators_) == len(experiment.param_grids_) == len(OVERSAMPLERS) * len(CLASSIFIERS) * n_runs
    

def test_datasets_summary():
    """Test the dataset's summary."""
    EXPERIMENT._summarize_datasets()
    expected_datasets_summary = pd.DataFrame(
        {
            'Dataset name': ['C', 'B', 'A'],
            'Features': [5, 10, 20],
            'Instances': [100, 100, 100],
            'Minority instances': [40, 20, 10],
            'Majority instances': [60, 80, 90],
            'Imbalance Ratio': [1.5, 4.0, 9.0]
        }
    )
    pd.testing.assert_frame_equal(EXPERIMENT.datasets_summary_, expected_datasets_summary, check_dtype=False)


def test_experiment_raised_errors():
    """Test if an attribute error is raised."""
    for func in ('_calculate_optimal_results', '_calculate_wide_optimal_results', '_calculate_ranking_results', 
                 '_calculate_mean_ranking_results', '_calculate_friedman_test_results', '_calculate_adjusted_pvalues_results'):
        with pytest.raises(AttributeError):
            getattr(EXPERIMENT, func)()

def test_results():
    """Test the results of experiment."""
    EXPERIMENT.run()
    assert EXPERIMENT.results_.reset_index().columns.get_level_values(0).tolist()[:-2] == GROUP_KEYS
    assert len(EXPERIMENT.results_) == len(DATASETS) * len(ParameterGrid(EXPERIMENT.param_grids_)) // EXPERIMENT.n_runs


def test_optimal():
    """Test the optimal results of experiment."""
    EXPERIMENT._calculate_optimal()
    assert set(EXPERIMENT.optimal_.Dataset.unique()) == set(EXPERIMENT.datasets_names_)
    assert set(EXPERIMENT.optimal_.Oversampler.unique()) == set(EXPERIMENT.oversamplers_names_)
    assert set(EXPERIMENT.optimal_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert len(EXPERIMENT.optimal_) == len(DATASETS) * len(OVERSAMPLERS) * len(CLASSIFIERS)


def test_wide_optimal():
    """Test the wide optimal results of experiment."""
    EXPERIMENT._calculate_wide_optimal()
    assert set(EXPERIMENT.wide_optimal_.Dataset.unique()) == set(EXPERIMENT.datasets_names_)
    assert set(EXPERIMENT.wide_optimal_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.wide_optimal_.columns)
    assert len(EXPERIMENT.wide_optimal_) == len(DATASETS) * len(CLASSIFIERS)


def test_ranking_results():
    """Test the ranking results of experiment."""
    EXPERIMENT._calculate_ranking()
    assert set(EXPERIMENT.ranking_.Dataset.unique()) == set(EXPERIMENT.datasets_names_)
    assert set(EXPERIMENT.ranking_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.ranking_.columns)
    assert len(EXPERIMENT.ranking_) == len(DATASETS) * len(CLASSIFIERS)

def test_mean_sem_ranking():
    """Test the mean ranking results of experiment."""
    EXPERIMENT._calculate_mean_sem_ranking()
    assert set(EXPERIMENT.mean_ranking_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.sem_ranking_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.mean_ranking_.columns)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.sem_ranking_.columns)
    assert len(EXPERIMENT.mean_ranking_) == len(CLASSIFIERS)
    assert len(EXPERIMENT.sem_ranking_) == len(CLASSIFIERS)


def test_mean_sem_scores():
    """Test the mean scores results of experiment."""
    EXPERIMENT._calculate_mean_sem_scores()
    assert set(EXPERIMENT.mean_scores_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.sem_scores_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.mean_scores_.columns)
    assert set(EXPERIMENT.oversamplers_names_).issubset(EXPERIMENT.sem_scores_.columns)
    assert len(EXPERIMENT.mean_scores_) == len(CLASSIFIERS)
    assert len(EXPERIMENT.sem_scores_) == len(CLASSIFIERS)


def test_mean_sem_perc_diff_scores():
    """Test the mean percentage difference of scores."""
    EXPERIMENT._calculate_mean_sem_perc_diff_scores(compared_oversamplers=None)
    assert set(EXPERIMENT.mean_perc_diff_scores_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.sem_perc_diff_scores_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert set(EXPERIMENT.mean_perc_diff_scores_.columns) == set(['Classifier', 'Metric', 'Difference'])
    assert set(EXPERIMENT.sem_perc_diff_scores_.columns) == set(['Classifier', 'Metric', 'Difference'])
    assert len(EXPERIMENT.mean_perc_diff_scores_) == len(CLASSIFIERS)
    assert len(EXPERIMENT.sem_perc_diff_scores_) == len(CLASSIFIERS)


def test_friedman_test():
    """Test the results of friedman test."""
    EXPERIMENT._calculate_friedman_test(alpha=0.05)
    assert set(EXPERIMENT.friedman_test_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert len(EXPERIMENT.friedman_test_) == len(CLASSIFIERS)


def test_holms_test():
    """Test the results of holms test."""
    EXPERIMENT._calculate_holms_test(control_oversampler=None)
    assert set(EXPERIMENT.holms_test_.Classifier.unique()) == set(EXPERIMENT.classifiers_names_)
    assert len(EXPERIMENT.holms_test_) == len(CLASSIFIERS)


def test_dump():
    """Test the dump method."""
    EXPERIMENT.dump()
    file_name = f'{EXPERIMENT.name}.pkl'
    with open(file_name, 'rb') as file:
        experiment = load(file)
        for attr_name in ATTR_NAMES:
            pd.testing.assert_frame_equal(getattr(EXPERIMENT, attr_name), getattr(experiment, attr_name))
    remove(file_name)
