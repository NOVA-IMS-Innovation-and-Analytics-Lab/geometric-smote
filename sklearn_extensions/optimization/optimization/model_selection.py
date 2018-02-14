"""
This module contains the class to compare pipelines
with various hyperparameters.
"""

from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.linear_model import LogisticRegression


class _ParametrizedEstimator(_BaseComposition):
    """Base class to parametrize any estimator to be used in the
    parameter grid search.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def set_params(self, **params):
        super()._set_params('estimator', **params)
        return self

    def get_params(self, deep=True):
        return super()._get_params('estimator', deep=deep)

    def fit(self, X, y, sample_weight=None):
        """Fit the estimators.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.
        """
        self.estimator_ = clone(self.estimator[0][1]).fit(X, y, sample_weight)
        return self

class _ParametrizedClassifier(_ParametrizedEstimator, ClassifierMixin):
    """A class that parametrizes any classifier to be used in the
    parameter grid search.

    Parameters
    ----------
    estimator : list of (string, classifier) tuple
        Invoking the fit method on the ParametrizedClassifier will fit
        a clone of the original classifier that will be stored in
        the class attribute self.estimator_.
    """

    def predict(self, X):
        """Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        """
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        """
        return self.estimator_.predict_proba(X)

class _ParametrizedRegressor(_ParametrizedEstimator, RegressorMixin):
    """A class that parametrizes any regressor to be used in the
    parameter grid search.

    Parameters
    ----------
    eor : list of (string, regressor) tuple
        Invoking the fit method on the ParametrizedClassifier will fit
        a clone of the original regressor that will be stored in
        the class attribute self.estimator_.
    """

    def predict(self, X):
        """Predict regression values for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        """
        return self.estimator_.predict(X)
    