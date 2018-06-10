"""
Test the imbalanced_analysis module.
"""

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification

X_clf, y_clf = make_classification()
CLASSIFIERS = [
    ('logr', LogisticRegression()),
    ('svc', SVC()),
    ('clf_pip', Pipeline([('scaler', MinMaxScaler()), ('lr', LogisticRegression())]))
]
CLASSIFIERS_PARAM_GRIDS = {'svc__C': [0.01, 0.1, 1.0], 'svc__kernel': ['rbf', 'linear']}