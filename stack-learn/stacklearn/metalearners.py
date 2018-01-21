"""
This module contains various metalearning estimators for classification and regression tasks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
from scipy.optimize import nnls
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_array
from .base import BaseMetaEstimator
from .metafeatures import CrossValidationExtractor
from .validation import check_all_estimators
from .utils import _parallel_map, _fit_estimator, _predict_estimator, _transform_probabilities, _predict_proabilities


class StackClassifier(BaseMetaEstimator, ClassifierMixin):
    """A general metaestimator for classification tasks.

    Parameters
    ----------
    meta_estimator :  estimator
        Invoking the `fit` method on the `StackClassifier` will fit a clone of 
        the meta_estimator on the transformed metalearning data.
    
    estimators : list of (string, estimator) tuples (optional)
        The base estimators of the stacked model. The estimators should 
        be classifiers.

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
    meta_estimator_ : a (string, estimator) named tuple
        The fitted metaestimator (a classifier).

    cve_ : CrossValidationExtractor object
        The CrossValidationExtractor object used to train the 
        base estimators.

    Examples
    --------
    >>> from stacklearn.metalearners import StackClassifier
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X, y = iris.data, iris.target
    >>> sc = StackClassifier()
    >>> sc.fit(X, y)
    StackClassifier(cv=None, estimators=None, meta_estimator=None, n_jobs=1)
    >>> print(sc.predict_proba(X[0:5]).shape)
    (5, 3)
    """

    FITTED_PARAMETER = "meta_estimator_"

    def __init__(self, estimators=None, meta_estimator=None, cv=None, n_jobs=1):
        super().__init__(estimators, cv, n_jobs)
        self.meta_estimator = meta_estimator
        
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

        X_transformed = super()._fit_transform(X, y)
        
        # Train the metaestimator
        self.meta_estimator_ = check_all_estimators(self.meta_estimator, self.cve_.estimators_, isinstance(self, ClassifierMixin), sample_weight)
        self.meta_estimator_ = _fit_estimator(self.meta_estimator_, X_transformed, y, sample_weight)

        return self

    def predict(self, X):
        """Predicts the class labels.
        
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

        X_transformed = self._transform(X)
        return self.meta_estimator_.model.predict(X_transformed)

    def predict_proba(self, X):
        """Predicts the class probabilities.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            
        Returns
        ----------
        predictions : array-like, shape = [n_samples, n_classes]
            Returns class probabilities where n_classes is the number 
            of classes.
        """

        X_transformed = self._transform(X)
        return self.meta_estimator_.model.predict_proba(X_transformed)


class StackRegressor(BaseMetaEstimator, RegressorMixin):
    """A general metaestimator for regression tasks.

    Parameters
    ----------
    meta_estimator :  estimator
        Invoking the `fit` method on the `StackRegressor` will fit a clone of 
        the meta_estimator on the transformed metalearning data.
    
    estimators : list of (string, estimator) tuples (optional)
        The base estimators of the stacked model. The estimators should 
        be regressors.

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
    meta_estimator_ : a (string, estimator) named tuple
        The fitted metaestimator (a regressor).

    cve_ : CrossValidationExtractor object
        The CrossValidationExtractor object used to train the 
        base estimators.

    Examples
    --------
    >>> from stacklearn.metalearners import StackRegressor
    >>> from sklearn import datasets
    >>> boston = datasets.load_boston()
    >>> X, y = boston.data, boston.target
    >>> sr = StackRegressor() 
    >>> sr.fit(X, y)
    StackRegressor(cv=None, estimators=None, meta_estimator=None, n_jobs=1)
    >>> print(sr.predict(X[0:10]).shape)
    (10,)
    """

    FITTED_PARAMETER = "meta_estimator_"

    def __init__(self, estimators=None, meta_estimator=None, cv=None, n_jobs=1):
        super().__init__(estimators, cv, n_jobs)
        self.meta_estimator = meta_estimator
        
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

        X_transformed = super()._fit_transform(X, y)
        
        # Train the metaestimator
        self.meta_estimator_ = check_all_estimators(self.meta_estimator, self.cve_.estimators_, isinstance(self, ClassifierMixin), sample_weight)
        self.meta_estimator_ = _fit_estimator(self.meta_estimator_, X_transformed, y, sample_weight)

        return self

    def predict(self, X):
        """Predicts the target values.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            
        Returns
        ----------
        predictions : array-like, shape = [n_samples]
            Returns the predicted target values.
        """

        X_transformed = self._transform(X)
        return self.meta_estimator_.model.predict(X_transformed)


class SuperLearnerClassifier(BaseMetaEstimator, ClassifierMixin):
    """The Super Learner classifier.

    Parameters
    ----------
    
    estimators : list of (string, estimator) tuples (optional)
        The base estimators of the Super Learner model.
        The estimators should be classifiers.

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
    weights_ : numpy array
        The optimal weights for the linear combination of base estimators.

    cve_ : CrossValidationExtractor object
        The CrossValidationExtractor object used to train the 
        base estimators.

    Examples
    --------
    >>> from stacklearn.metalearners import SuperLearnerClassifier
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> X, y = iris.data, iris.target
    >>> slc = SuperLearnerClassifier() 
    >>> slc.fit(X, y)
    SuperLearnerClassifier(cv=None, estimators=None, n_jobs=1)
    >>> print(slc.predict_proba(X[0:10]).shape)
    (10, 3)
    """
    
    FITTED_PARAMETER = "weights_"

    def fit(self, X, y):
        """Fit the base estimators and optimize the weights.
    
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """

        X_transformed = super()._fit_transform(X, y)
        if self.cve_.type_of_target_ == "continuous":
            raise TypeError("Super Learner classifier is not allowed for a continuous target.")
        self.classes_ = np.unique(y)
        
        # Transform to log loss
        indices_array = np.digitize(y, self.classes_).reshape(-1, 1) - 1
        class_probabilities = _transform_probabilities(X_transformed, indices_array, len(self.cve_.estimators_), len(self.classes_))
        
        # Optimize and normalize weights
        self.weights_, _ = nnls(class_probabilities, y)
        self.weights_ = self.weights_ / self.weights_.sum()

        return self

    def predict(self, X):
        """Predicts the class labels.
        
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

        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def predict_proba(self, X):
        """Predicts the class probabilities.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            
        Returns
        ----------
        predictions : array-like, shape = [n_samples, n_classes]
            Returns the predicted probabilities where n_classes is the number 
            of classes.
        """

        X_transformed = self._transform(X)
        return _predict_proabilities(X_transformed, self.weights_, len(self.cve_.estimators_), len(self.classes_))


class SuperLearnerRegressor(BaseMetaEstimator, RegressorMixin):
    """The Super Learner regressor.

    Parameters
    ----------
    
    estimators : list of (string, estimator) tuples (optional)
        The base estimators of the Super Learner model.
        The estimators should be regressors.

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
    weights_ : numpy array
        The optimal weights for the linear combination of base estimators.

    cve_ : CrossValidationExtractor object
        The CrossValidationExtractor object used to train the 
        base estimators.

    Examples
    --------
    >>> from stacklearn.metalearners import SuperLearnerRegressor
    >>> from sklearn import datasets
    >>> boston = datasets.load_boston()
    >>> X, y = boston.data, boston.target
    >>> slr = SuperLearnerRegressor() 
    >>> slr.fit(X, y)
    SuperLearnerRegressor(cv=None, estimators=None, n_jobs=1)
    >>> print(slr.predict(X[0:10]).shape)
    (10,)
    """
    
    FITTED_PARAMETER = "weights_"

    def fit(self, X, y):
        """Fit the base estimators and optimize the weights.
    
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
            
        Returns
        -------
        self : object
            Returns self.
        """

        X_transformed = super()._fit_transform(X, y)
        if self.cve_.type_of_target_ != "continuous":
            raise TypeError("Super Learner regressor is not allowed for a binary or multiclass target.")
        
        # Optimize and normalize weights
        self.weights_, _ = nnls(X_transformed, y)
        self.weights_ = self.weights_ / self.weights_.sum()

        return self

    def predict(self, X):
        """Predicts the target values.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            
        Returns
        ----------
        predictions : array-like, shape = [n_samples]
            Returns the predicted target values.
        """

        return np.dot(self._transform(X), self.weights_)