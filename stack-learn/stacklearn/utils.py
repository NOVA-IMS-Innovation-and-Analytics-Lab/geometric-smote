"""Helper functions."""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
from sklearn.externals.joblib import Parallel, delayed


def _parallel_map(n_jobs, map_function, map_tasks, **kwargs):
        """Wrapper function for Parallel class"""

        return Parallel(n_jobs)(delayed(map_function)(map_task, **kwargs) for map_task in map_tasks)

def _fit_estimator(estimator, X, y, sample_weight=None):
    """Private function that fits a clone of an estimator in each task."""

    if sample_weight:
        estimator.model.fit(X, y, sample_weight)
    else:
        estimator.model.fit(X, y)
    return estimator

def _predict_estimator(estimator, X, predict_probability=False):
    """Private function that predicts the target or the class 
    labels in each task."""

    if predict_probability:
        predictions = estimator.model.predict_proba(X)
    else:
        predictions = estimator.model.predict(X).reshape(-1, 1)
    return predictions

def _choose_elements(row, n_estimators, n_classes):
    """Private function that chooses the appropiate elements of a 
    row for the super learner classifier tranformed matrix."""
    
    class_index = row.size - 1
    class_label = int(row[class_index])
    class_indices = [class_label + n_classes * step for step in range(n_estimators)]
    return row[class_indices]

def _transform_probabilities(X_transformed, indices_array, n_estimators, n_classes):
    """Private function that transforms the input matrix of super 
    learner classifier to class probabilities for the predicted class."""

    X_transformed = np.hstack([X_transformed, indices_array])
    return np.apply_along_axis(_choose_elements, 1, X_transformed, n_estimators=n_estimators, n_classes=n_classes)

def _predict_proabilities(X_transformed, weights, n_estimators, n_classes):
    """"Private function that uses the optimal weights of the super 
    learner classifier to predict the class probabilities."""
    
    for cl in range(n_classes):
        class_indices = [cl + n_classes * step for step in range(n_estimators)]
        class_probabilities = np.dot(X_transformed[:, class_indices], weights).reshape(-1, 1)
        predicted_probabilities = np.hstack([predicted_probabilities, class_probabilities]) if cl > 0 else class_probabilities 
    return predicted_probabilities