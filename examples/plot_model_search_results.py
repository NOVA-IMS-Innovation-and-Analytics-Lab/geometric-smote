#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===========================
Model Search Report Example
===========================

"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from sklearnext.cluster import KMeans
from sklearnext.model_selection import ModelSearchCV
from sklearnext.over_sampling import SMOTE
from sklearnext.over_sampling.base import DensityDistributor
from sklearnext.tools import report_model_search_results

# Load data
X, y = make_classification(n_informative=15, n_clusters_per_class=3, weights=[0.9, 0.1])

# Define estimators
estimators = [
    ('GBC', GradientBoostingClassifier()), 
    ('SMOTE+GBC', Pipeline([('smote', SMOTE()), ('gbc', GradientBoostingClassifier())])),
    ('KMeanSMOTE+GBC', Pipeline([('smote', SMOTE(clusterer=KMeans(n_init=1), distributor=DensityDistributor())), ('gbc', GradientBoostingClassifier())]))
]

# Define parameters grid
param_grids = [
    {'SMOTE+GBC__smote__k_neighbors': [2, 3, 4, 5], 'SMOTE+GBC__gbc__max_depth': [2, 4]}, 
    {
        'KMeanSMOTE+GBC__smote__k_neighbors': [2, 3, 4, 5],
        'KMeanSMOTE+GBC__smote__clusterer__n_clusters': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0], 
        'KMeanSMOTE+GBC__smote__distributor__filtering_threshold': [0.8, 1.0, 1.2, 1.5],
        'KMeanSMOTE+GBC__gbc__max_depth': [2, 4]
    }
]

# Define model search object
model_search_cv = ModelSearchCV(
    estimators=estimators, 
    param_grids=param_grids, 
    scoring=['geometric_mean_score', 'roc_auc'], 
    cv=StratifiedKFold(n_splits=5, shuffle=True),
    refit=False, 
    n_jobs=-1
)

# Fit model search
model_search_cv.fit(X, y)

# Print results
print(__doc__, report_model_search_results(model_search_cv, sort_results='mean_test_geometric_mean_score'))
