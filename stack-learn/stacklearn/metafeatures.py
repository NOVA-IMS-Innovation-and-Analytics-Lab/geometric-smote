"""
This module contains the CrossValidationExtractor class that transforms an input matrix 
based on the predictions of a set of estimators.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import check_cv
from .validation import check_base_estimators, check_type_of_target
from .utils import _parallel_map, _fit_estimator, _predict_estimator
from collections import namedtuple
from itertools import product

task_output = namedtuple("task_output", ["predictions", "indices"])

class CrossValidationExtractor(TransformerMixin):

    """Class that creates a feature extractor using cross validation and
    a set of base estimators.  
    
    Parameters
    ----------
    estimators : list of (string, estimator) tuples (optional)
        Invoking the `fit` method on the `CrossValidationExtractor` will
        fit clones of these estimators.

    cv : int, cross-validation generator or an iterable (optional)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

    n_jobs : int (optional, default 1)
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    Attributes
    ----------
    estimators_ : list of (string, estimator) named tuples
        A list of the fitted base estimators.

    cv_ : int, cross-validation generator or an iterable

    type_of_target_ : str
        The type of target.

    Notes
    -----
    The extracted features for any classifier are always class probabilities even if 
    probability=False. Regressors can be used for any type of target. 

    Examples
    --------
    >>> from stacklearn.metafeatures import CrossValidationExtractor
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X, y = iris.data, iris.target
    >>> cve = CrossValidationExtractor()
    >>> X_transformed = cve.fit_transform(X, y)
    >>> print(X.shape)
    (150, 4)
    >>> print(X_transformed.shape)
    (150, 12)
    """

    def __init__(self, estimators=None, cv=None, n_jobs=1):
        self.estimators = estimators
        self.cv = cv
        self.n_jobs = n_jobs

    @staticmethod
    def _task_map_function(map_task, **kwargs):
        """Private static method that fits an estimator on the train indices 
        and returns predictions on the test indices and the test indices."""

        # Unpack values
        estimator, indices = map_task
        train_indices, test_indices = indices
        X, y, type_of_target = map(kwargs.get, ("X", "y", "type_of_target"))

        # Fit the training data to the estimator
        _fit_estimator(estimator, X[train_indices], y[train_indices])

        # For a classifier predict class probabilities and for a regressor predict the target
        predictions = _predict_estimator(estimator, X[test_indices], predict_probability=type_of_target is not "continuous")
            
        return task_output(predictions, test_indices)

    def _transform_parallel_output(self, output):
        """Private method that transforms the output of the Parallel function."""

        # Horizontal reshape of output
        for task_ind, task_output in enumerate(output):
            X_transformed_horizontal = np.vstack([X_transformed_horizontal, task_output.predictions]) if task_ind > 0 else task_output.predictions
            if task_ind < self.cv_.n_splits:
                indices = np.append(indices, task_output.indices) if task_ind > 0 else task_output.indices
        
        # Vertical reshape of output
        for ind, X_transformed_vertical in enumerate(np.split(X_transformed_horizontal, len(self.estimators_))):
            X_transformed = np.hstack([X_transformed, X_transformed_vertical]) if ind > 0 else X_transformed_vertical
        
        # Sort final array
        X_transformed = X_transformed[np.argsort(indices)]

        return X_transformed

    def fit_transform(self, X, y):
        """Fit the base estimators to data applying cross validation
        and then transform it predicting the validation sets as inputs 
        for each base estimator and cross validation stage.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        X_transformed : numpy array of shape [n_samples, n_estimators * n_predicted_features]
            Target values or class probabilities predicted by each estimator.
        """

        # Check input data
        X, y = check_X_y(X, y)

        # Set type of target, number of classes and number of samples
        self.type_of_target_, self.n_classes_ = check_type_of_target(y)
        self.n_samples_ = X.shape[0]

        # Check cv and estimators
        self.estimators_ = check_base_estimators(self.estimators, self.type_of_target_)
        self.cv_ = check_cv(self.cv, y, self.type_of_target_ is not "continuous")
               
        # Return data of extracted features
        map_function =  CrossValidationExtractor._task_map_function
        map_tasks = product(self.estimators_, self.cv_.split(X=X, y=y))
        output = _parallel_map(self.n_jobs, map_function, map_tasks, X=X, y=y, type_of_target = self.type_of_target_)
        
        return self._transform_parallel_output(output)