"""
The :mod:`sklearnext.utils.estimators` includes
various helper estimators and oversamplers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import re
from warnings import warn, filterwarnings
from dask_searchcv.utils import copy_estimator
from sklearn.metrics import r2_score, accuracy_score
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
import progressbar
from ..utils import check_estimators


class _ParametrizedEstimators(_BaseComposition):
    """The functionality of a collection of estimators is provided as
    a single metaestimator. The fitted estimator is selected using a
    parameter."""

    def __init__(self, estimators, est_name=None, random_state=None):
        self.estimators = estimators
        self.est_name = est_name
        self.random_state = random_state
        check_estimators(estimators)
        self._validate_names([est_name for est_name, _ in estimators])
        _ParametrizedEstimators._estimator_type = self._return_estimator_type()

    def _return_estimator_type(self):
        _, steps = zip(*self.estimators)
        if len(set([step._estimator_type for step in steps if hasattr(step, '_estimator_type')])) > 1:
            warn('Estimators include both regressors and classifiers. Estimator type set to classifier.')
            return 'classifier'
        return steps[0]._estimator_type

    @classmethod
    def _create_progress_bar(cls, n_fitting_tasks):
        cls.progress_bar = progressbar.ProgressBar(max_value=n_fitting_tasks, widgets=[progressbar.Percentage()])
        if not hasattr(cls, 'n_fitting_tasks'):
            cls.n_fitting_tasks = 1
        if all([hasattr(cls, attribute) for attribute in ['ind', 'dataset_name', 'n_datasets']]):
            cls.progress_bar.max_value *= cls.n_datasets
            cls.progress_bar.prefix = 'Current dataset: {} | Completed datasets: {}/{} | Progress: '.format(
                cls.dataset_name, cls.ind, cls.n_datasets)
        else:
            cls.progress_bar.prefix = 'Progress: '

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction
        if estimator type is a regressor or the mean accuracy on the given
        test data and labels if estimator type is a classifier.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        if _ParametrizedEstimators._estimator_type == 'regressor':
            score = r2_score(y, self.predict(X), sample_weight=sample_weight, multioutput='variance_weighted')
        elif _ParametrizedEstimators._estimator_type == 'classifier':
            score = accuracy_score(y, self.predict(X), sample_weight=sample_weight)
        return score

    def set_params(self, **params):
        """Set the parameters.
        Valid parameter keys can be listed with get_params().
        Parameters
        ----------
        params : keyword arguments
            Specific parameters using e.g. set_params(parameter_name=new_value)
            In addition, to setting the parameters of the ``_ParametrizedEstimators``,
            the individual estimators of the ``_ParametrizedEstimators`` can also be
            set or replaced by setting them to None.
        """
        super()._set_params('estimators', **params)
        check_estimators(self.estimators)
        return self

    def get_params(self, deep=True):
        """Get the parameters.
        Parameters
        ----------
        deep: bool
            Setting it to True gets the various estimators and the parameters
            of the estimators as well
        """
        return super()._get_params('estimators', deep=deep)

    def fit(self, X, y, *args, **kwargs):
        """"Fit the selected estimator and dataset."""

        # Copy one of the estimators
        if self.est_name is None:
            raise ValueError('Attribute `est_name` is set to None. An estimator should be selected.')
        estimator = copy_estimator(dict(self.estimators)[self.est_name])

        # Fix data race
        filterwarnings('ignore', category=DeprecationWarning, module=r'^{0}\.'.format(re.escape(__name__)))

        # Set random state when exists
        params = estimator.get_params().keys()
        random_state_params = [par for par, included in zip(params, ['random_state' in par for par in params]) if included]
        for par in random_state_params:
            estimator.set_params(**{par: self.random_state})

        # Fit estimator
        self.estimator_ = estimator.fit(X, y, *args, **kwargs)

        # Increase number of fitted tasks
        if hasattr(_ParametrizedEstimators, 'progress_bar'):
            _ParametrizedEstimators.progress_bar.update(_ParametrizedEstimators.n_fitting_tasks)
            _ParametrizedEstimators.n_fitting_tasks += 1

        return self

    def predict(self, X, *args, **kwargs):
        """"Predict with the selected estimator."""
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        """"Predict the probability with the selected estimator."""
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict_proba(X, *args, **kwargs)