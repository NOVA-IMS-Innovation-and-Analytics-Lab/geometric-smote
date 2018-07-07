"""
The :mod:`sklearnext.utils.estimators` includes
various helper estimators and oversamplers.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import re
from warnings import filterwarnings
from dask_searchcv.utils import copy_estimator
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
import progressbar
from ..utils import check_estimators


class _ParametrizedEstimatorsMixin(_BaseComposition):
    """Mixin class for all parametrized estimators."""

    def __init__(self, estimators, est_name=None, random_state=None):
        self.estimators = estimators
        self.est_name = est_name
        self.random_state = random_state
        check_estimators(estimators)
        self._validate_names([est_name for est_name, _ in estimators])

    @classmethod
    def _create_progress_bar(cls, n_fitting_tasks, scheduler):
        cls.tasks_text = progressbar.FormatCustomText('(%(tasks)d / %(n_tasks)d)', dict(tasks=0, n_tasks=n_fitting_tasks))
        cls.progress_bar = progressbar.ProgressBar(max_value=n_fitting_tasks,
                                                   widgets=[
                                                       progressbar.Percentage(),
                                                       ' ',
                                                       cls.tasks_text])
        cls.fitting_task = 1
        progress_bar_msg = str() if scheduler is 'multiprocessing' else 'Progress: '
        if all([hasattr(cls, attribute) for attribute in ['ind', 'dataset_name', 'n_datasets']]):
            progress_bar_msg = 'Current dataset: {} | Completed datasets: {}/{} | ' + progress_bar_msg
            cls.progress_bar.prefix = progress_bar_msg.format(cls.dataset_name, cls.ind, cls.n_datasets)
        else:
            cls.progress_bar.prefix = progress_bar_msg

    def set_params(self, **params):
        """Set the parameters.
        Valid parameter keys can be listed with get_params().
        Parameters
        ----------
        params : keyword arguments
            Specific parameters using e.g. set_params(parameter_name=new_value)
            In addition, to setting the parameters of the ``_ParametrizedEstimatorsMixin``,
            the individual estimators of the ``_ParametrizedEstimatorsMixin`` can also be
            set or replaced by setting them to None.
        """
        super(_ParametrizedEstimatorsMixin, self)._set_params('estimators', **params)
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
        return super(_ParametrizedEstimatorsMixin, self)._get_params('estimators', deep=deep)

    def fit(self, X, y, *args, **kwargs):
        """"Fit the selected estimator."""

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

        return self

    def predict(self, X, *args, **kwargs):
        """"Predict with the selected estimator."""
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(X, *args, **kwargs)


class _ParametrizedClassifiers(_ParametrizedEstimatorsMixin, ClassifierMixin):
    """The functionality of a collection of classifiers is provided as
    a single metaclassifier. The classifier to be fitted is selected using a
    parameter."""

    def fit(self, X, y, *args, **kwargs):
        """"Fit the selected classifier."""

        super(_ParametrizedClassifiers, self).fit(X, y, *args, **kwargs)

        # Increase number of fitted tasks
        if hasattr(_ParametrizedClassifiers, 'progress_bar'):
            _ParametrizedClassifiers.progress_bar.update(_ParametrizedClassifiers.fitting_task)
            _ParametrizedClassifiers.tasks_text.update_mapping(tasks=_ParametrizedClassifiers.fitting_task)
            _ParametrizedClassifiers.fitting_task += 1

        return self

    def predict_proba(self, X, *args, **kwargs):
        """"Predict the probability with the selected estimator."""
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict_proba(X, *args, **kwargs)


class _ParametrizedRegressors(_ParametrizedEstimatorsMixin, RegressorMixin):
    """The functionality of a collection of regressors is provided as
    a single metaregressor. The regressor to be fitted is selected using a
    parameter."""

    def fit(self, X, y, *args, **kwargs):
        """"Fit the selected regressor."""

        super(_ParametrizedRegressors, self).fit(X, y, *args, **kwargs)

        # Increase number of fitted tasks
        if hasattr(_ParametrizedRegressors, 'progress_bar'):
            _ParametrizedRegressors.progress_bar.update(_ParametrizedRegressors.fitting_task)
            _ParametrizedRegressors.tasks_text.update_mapping(tasks=_ParametrizedRegressors.fitting_task)
            _ParametrizedRegressors.fitting_task += 1

        return self

