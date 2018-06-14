"""
Test the imbalanced_analysis module.
"""

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from ..imbalanced_analysis import evaluate_imbalanced_experiments

X1, y1 = make_classification(weights=[0.90, 0.10], n_samples=100)
X2, y2 = make_classification(weights=[0.70, 0.30], n_samples=80)
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


@pytest.mark.parametrize('scoring,n_runs', [
    ('accuracy', 3)
])
def test_evaluate_imbalanced_experiment(scoring, n_runs):
    """Test the output of the model search report function."""
    evaluation = evaluate_imbalanced_experiments(DATASETS, OVERSAMPLERS, CLASSIFIERS, scoring, n_runs=n_runs)
