"""
The :mod:`sklearnext.model_selection.search` includes utilities to search
the parameter and model space.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV

from ..utils import check_estimator_type, check_param_grids, check_estimators


class MultiEstimatorMixin(_BaseComposition):
    """Mixin class for multi estimator."""

    def __init__(self, estimators, est_name=None, random_state=None):
        
        check_estimators(estimators)      
        self.estimators = estimators
        self.est_name = est_name
        self.random_state = random_state
        self._validate_names([est_name for est_name, _ in estimators])

    def set_params(self, **params):
        """Set the parameters.
        Valid parameter keys can be listed with get_params().
        Parameters
        ----------
        params : keyword arguments
            Specific parameters using e.g. set_params(parameter_name=new_value)
            In addition, to setting the parameters of the ``MultiEstimatorMixin``,
            the individual estimators of the ``MultiEstimatorMixin`` can also be
            set or replaced by setting them to None.
        """
        super(MultiEstimatorMixin, self)._set_params('estimators', **params)
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
        return super(MultiEstimatorMixin, self)._get_params('estimators', deep=deep)

    def fit(self, X, y, **fit_params):
        """"Fit the selected estimator."""

        # Copy one of the estimators
        if self.est_name is None:
            raise ValueError('Attribute `est_name` is set to None. An estimator should be selected.')
        estimator = clone(dict(self.estimators)[self.est_name])

        # Set random state when exists
        params = estimator.get_params().keys()
        random_state_params = [par for par, included in zip(params, ['random_state' in par for par in params]) if included]
        for par in random_state_params:
            estimator.set_params(**{par: self.random_state})

        # Fit estimator
        self.estimator_ = estimator.fit(X, y, **fit_params)

        return self

    def predict(self, X, *args, **kwargs):
        """"Predict with the selected estimator."""
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict(X, *args, **kwargs)


class MultiClassifier(MultiEstimatorMixin, ClassifierMixin):
    """The functionality of a collection of classifiers is provided as
    a single metaclassifier. The classifier to be fitted is selected using a
    parameter."""

    def predict_proba(self, X, *args, **kwargs):
        """"Predict the probability with the selected estimator."""
        check_is_fitted(self, 'estimator_')
        return self.estimator_.predict_proba(X, *args, **kwargs)


class MultiRegressor(MultiEstimatorMixin, RegressorMixin):
    """The functionality of a collection of regressors is provided as
    a single metaregressor. The regressor to be fitted is selected using a
    parameter."""

    pass


class ModelSearchCV(GridSearchCV):
    """Exhaustive search over specified model and parameter values for a collection of estimators.

    Important members are fit, predict.

    ModelSearchCV implements a "fit" and a "score" method.
    It also implements "predict", "predict_proba", "decision_function",
    "transform" and "inverse_transform" if they are implemented in the
    estimators used.

    The parameters of the estimators used to apply these methods are optimized
    by cross-validated grid-search over their parameter grids.

    Parameters
    ----------
    estimators :  list of (string, estimator) tuples
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grids : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) strings
        or a dict with names as keys and callables as values.

        NOTE that when using custom scorers, each scorer should return a single
        value. Metric functions returning a list/array of values can be wrapped
        into multiple scorers that return one value each.

        If None, a default scorer is used.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    iid : boolean, default=True
        If True, return the average score across folds, weighted by the number
        of samples in each test set. In this case, the data is assumed to be
        identically distributed across the folds, and the loss minimized is
        the total loss per sample, and not the mean loss across the folds. If
        False, return the average score across folds.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.

    refit : boolean, or string, default=True
        Refit an estimator using the best found parameters on the whole
        dataset.

        For multiple metric evaluation, this needs to be a string denoting the
        scorer is used to find the best parameters for refitting the estimators
        at the end.

        The refitted estimators is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``ModelSearchCV`` instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : boolean, optional (default=True)
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.


    Examples
    --------
    >>> from sklearn import svm, datasets
    >>> from sklearn.model_selection import GridSearchCV
    >>> iris = datasets.load_iris()
    >>> parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    >>> svc = svm.SVC(gamma="scale")
    >>> clf = GridSearchCV(svc, parameters, cv=5)
    >>> clf.fit(iris.data, iris.target)
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    GridSearchCV(cv=5, error_score=...,
           estimator=SVC(C=1.0, cache_size=..., class_weight=..., coef0=...,
                         decision_function_shape='ovr', degree=..., gamma=...,
                         kernel='rbf', max_iter=-1, probability=False,
                         random_state=None, shrinking=True, tol=...,
                         verbose=False),
           fit_params=None, iid=..., n_jobs=None,
           param_grid=..., pre_dispatch=..., refit=..., return_train_score=...,
           scoring=..., verbose=...)
    >>> sorted(clf.cv_results_.keys())
    ...                             # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ['mean_fit_time', 'mean_score_time', 'mean_test_score',...
     'mean_train_score', 'param_C', 'param_kernel', 'params',...
     'rank_test_score', 'split0_test_score',...
     'split0_train_score', 'split1_test_score', 'split1_train_score',...
     'split2_test_score', 'split2_train_score',...
     'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score'...]

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +------------+-----------+------------+-----------------+---+---------+
        |param_kernel|param_gamma|param_degree|split0_test_score|...|rank_t...|
        +============+===========+============+=================+===+=========+
        |  'poly'    |     --    |      2     |       0.80      |...|    2    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'poly'    |     --    |      3     |       0.70      |...|    4    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.1   |     --     |       0.80      |...|    3    |
        +------------+-----------+------------+-----------------+---+---------+
        |  'rbf'     |     0.2   |     --     |       0.93      |...|    1    |
        +------------+-----------+------------+-----------------+---+---------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
                                         mask = [False False False False]...)
            'param_gamma': masked_array(data = [-- -- 0.1 0.2],
                                        mask = [ True  True False False]...),
            'param_degree': masked_array(data = [2.0 3.0 -- --],
                                         mask = [False False  True  True]...),
            'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
            'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
            'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
            'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
            'rank_test_score'    : [2, 4, 3, 1],
            'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
            'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
            'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
            'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
            'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
            'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
            'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    ------
    The parameters selected are those that maximize the score of the left out
    data, unless an explicit score is passed in which case it is used instead.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    point in the grid (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    """

    def __init__(self, estimators, param_grids, scoring=None,
                 n_jobs=None, iid=True, refit=True, cv=5, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise',
                 return_train_score=True):
        self.estimators = estimators
        self.param_grids = param_grids
        estimator = MultiClassifier(estimators) \
            if check_estimator_type(estimators) == 'classifier' \
            else MultiRegressor(estimators)
        super(ModelSearchCV, self).__init__(estimator=estimator,
                                            param_grid=check_param_grids(param_grids, estimators),
                                            scoring=scoring,
                                            n_jobs=n_jobs,
                                            iid=iid,
                                            refit=refit,
                                            cv=cv,
                                            verbose=verbose,
                                            pre_dispatch=pre_dispatch,
                                            error_score=error_score,
                                            return_train_score=return_train_score)

    @staticmethod
    def _split_est_name(param_grid):
        """Split the estimator name."""

        # Exclude random state
        param_grid = {param:value for param, value in param_grid.items() if param != 'random_state'}

        # Remove and get the estimator name
        est_name = param_grid.pop('est_name')

        return est_name, {'__'.join(param.split('__')[1:]):value for param, value in param_grid.items()}

    def _modify_grid_search_attrs(self):
        """Modify the object's grid search attributes."""

        # Create best estimator attribte
        if hasattr(self, 'best_estimator_'):
            self.best_estimator_ = self.best_estimator_.estimator_
        
        # Populate models list
        models = []
        for ind, param_grid in enumerate(self.cv_results_['params']):
            est_name, self.cv_results_['params'][ind] = self._split_est_name(param_grid)
            models.append(est_name)
        
        # Append models list to results
        self.cv_results_.update({'models': models})

    def fit(self, X, y=None, groups=None, **fit_params):

        # Call superclass fit method
        super(ModelSearchCV, self).fit(X, y, groups, **fit_params)
        
        # Modify attributes
        self._modify_grid_search_attrs()

        return self

