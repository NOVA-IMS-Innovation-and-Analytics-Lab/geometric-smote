#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===========================
Model Search Report Example
===========================

"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearnext.cluster import SOM
from sklearnext.model_selection import ModelSearchCV
from sklearnext.over_sampling import SMOTE
from sklearnext.over_sampling.base import DensityDistributor
from sklearnext.tools import report_model_search_results
from imblearn.pipeline import Pipeline

print(__doc__)

# Load data
dataset = load_breast_cancer()
X, y = dataset.data, dataset.target

# Define estimators
estimators = [
    ('LR', LogisticRegression()), 
    ('DT', DecisionTreeClassifier()),
    ('SOMO+GBC', Pipeline([('smote', SMOTE(clusterer=SOM(), distributor=DensityDistributor())), ('gbc', GradientBoostingClassifier())]))
]

# Define parameters grid
param_grids = [
    {'DT__max_depth': [2, 3, 4, 5], 'DT__criterion': ['gini', 'entropy']}, 
    {
        'SOMO+GBC__smote__k_neighbors': [3, 4, 5], 
        'SOMO+GBC__smote__clusterer__n_rows': [2, 4, 8, 10], 
        'SOMO+GBC__smote__clusterer__n_columns': [2, 4, 8, 10], 
        'SOMO+GBC__smote__distributor__filtering_threshold': [1.0, 1.5, 2.0], 
        'SOMO+GBC__gbc__max_depth': [3, 5, 8]
    }
]

# Define model search object
model_search_cv = ModelSearchCV(estimators=estimators, param_grids=param_grids, scoring=['accuracy', 'geometric_mean_score', 'f1'], refit=False)

# Fit model search
model_search_cv.fit(X, y)
