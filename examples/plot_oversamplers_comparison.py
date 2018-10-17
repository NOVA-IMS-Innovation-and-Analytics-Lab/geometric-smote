#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
================================
Over-sampling Comparison Example
================================

"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

# Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearnext.tools import evaluate_binary_imbalanced_experiments, summarize_binary_datasets
from sklearnext.over_sampling import SMOTE, GeometricSMOTE

# Oversamplers and classifiers
oversamplers = [
    ('NO OVERSAMPLING', None),
    ('SMOTE', SMOTE(random_state=0), {'k_neighbors':[3, 4]}),
    ('G-SMOTE', GeometricSMOTE(random_state=0), {
        'k_neighbors':[3, 4], 
        'deformation_factor': [0.25, 0.50, 0.75], 
        'truncation_factor': [-0.5, 0.0, 0.5]
        }
    )
]
classifiers = [
    ('DT', DecisionTreeClassifier(), {'max_depth': [3, 4, 5]}),
    ('KNN', KNeighborsClassifier(), {'n_neighbors':[3, 5]}),
]

# Generate datasets
imbalanced_datasets = [
    ('A', make_classification(random_state=1, weights=[0.80, 0.20], n_features=10)),
    ('B', make_classification(random_state=1, weights=[0.85, 0.15], n_features=10)),
    ('C', make_classification(random_state=1, weights=[0.90, 0.10], n_features=10))
]

# Summarize datasets
imbalanced_datasets_summary = summarize_binary_datasets(imbalanced_datasets)

# Compare oversamplers and classifiers
results = evaluate_binary_imbalanced_experiments(datasets=imbalanced_datasets,
                                                 oversamplers=oversamplers,
                                                 classifiers=classifiers,
                                                 scoring=['roc_auc', 'f1', 'geometric_mean_score'],
                                                 n_splits=5,
                                                 n_runs=2,
                                                 random_state=5)

# Extract results
scores, ranking = results['wide_optimal'], results['mean_ranking']

# Print datasets summary
print('\n\nSummary of datasets:')
print(imbalanced_datasets_summary)

# Print experiment results
print('\nScores for all combinations of datasets, oversamplers and classifiers:')
print(scores)
print('\nMean ranking of oversamplers across datasets:') 
print(ranking)
