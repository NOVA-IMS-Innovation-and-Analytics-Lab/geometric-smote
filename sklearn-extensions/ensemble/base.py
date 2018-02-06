"""
Abstract classes for metaestimators.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array
from .metafeatures import CrossValidationExtractor
from .utils import _parallel_map, _fit_estimator, _predict_estimator


class BaseMetaEstimator(BaseEstimator, metaclass=ABCMeta):

    def __init__(self, estimators=None, cv=None, n_jobs=1):
        self.estimators = estimators
        self.cv = cv
        self.n_jobs = n_jobs

    def _transform(self, X):
        """Private method that transforms any input matrix."""

        # Check the input data and if metaestimator is fitted
        check_is_fitted(self, self.FITTED_PARAMETER)
        X = check_array(X)

        # Return data of extracted features
        output = _parallel_map(self.n_jobs, _predict_estimator, self.cve_.estimators_, X=X, predict_probability=self.cve_.type_of_target_ is not "continuous")

        # Tranform the data
        for task_index, task_output in enumerate(output):
            X_transformed = np.hstack([X_transformed, task_output]) if task_index > 0 else task_output 
        return X_transformed
    
    def _fit_transform(self, X, y):
        """Private method that fits the base estimators and returns 
        the transformed input matrix.
    
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
            
        Returns
        -------
        X_transformed : numpy array of shape [n_samples, n_estimators * n_predicted_features]
            Returns the transformed input matrix.
        """

        # Create CrossValidationExtractor object and transform the input data
        self.cve_ = CrossValidationExtractor(self.estimators, self.cv, self.n_jobs)
        X_transformed = self.cve_.fit_transform(X, y)

        # Train the base estimators
        self.cve_.estimators_ = _parallel_map(self.n_jobs, _fit_estimator, self.cve_.estimators_, X=X, y=y)

        return X_transformed
    
    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the base estimators and the metalearner.
    
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            
        Returns
        -------
        self : object
            Returns self.
        """

        pass

    @abstractmethod
    def predict(self, X):
        """Predicts the class labels or target values.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            
        Returns
        ----------
        predictions : array-like, shape = [n_samples]
            Returns the predicted class labels.
        """

        pass