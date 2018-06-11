"""
Test the imbalanced_analysis module.
"""

import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

X, y = make_classification(weights=[0.90, 0.10])
OVERSAMPLERS = [
    ('random', RandomOverSampler()),
    ('smote', SMOTE(), {'k_neighbors': [2, 3, 4], 'kind': ['regular', 'borderline1']}),
    ('adasyn', ADASYN(), {'n_neighbors': [2, 3, 4, 5]})
]
CLASSIFIERS = [
    ('lr', LogisticRegression()),
    ('svc', SVC(), {'C': [0.1, 0.5, 1.0], 'kernel': ['rbf', 'linear']})
]