"""
This module contains classes to compare and evaluate
the performance of various oversampling algorithms.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from warnings import filterwarnings
from itertools import product
from pickle import dump
from progressbar import ProgressBar
from sklearn.model_selection import check_cv, GridSearchCV
from imblearn.pipeline import Pipeline
from .utils import check_datasets, check_random_states
from .metrics import SCORERS


class Experiment:
    """Class for comparison of a various transformations/estimators 
    pipelines across multiple datasets.

    Parameters
    ----------
    datasets : list of (dataset name, (X, y)) tuples
        The dataset name is a string and (X, y) are tuples of input data and
        target values.
    estimators : list of (estimator name, estimators) tuples
        Each estimator is a single estimator or a pipeline of transformations/estimator.
    param_grids : list of grids
        Each grid corresponds to the parameters of a pipeline.
    scoring : string, callable, list/tuple, dict or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        to evaluate the predictions on the test set. For evaluating multiple 
        metrics, either give a list of (unique) strings or a dict 
        with names as keys and callables as values.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross validation.
        - integer, to specify the number of folds in a `(Stratified)KFold`.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.
    experiment_repetitions : int, (default=5)
        The number of experiment repetitions.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    n_jobs : int, (default=1)
        The number of CPUs to use to do the computation. -1 means ‘all CPUs’.
    """

    def __init__(self,
                 datasets,
                 estimators,
                 param_grids,
                 scoring=['roc_auc', 'f1', 'geometric_mean_score'],
                 cv=None,
                 experiment_repetitions=5,
                 random_state=None,
                 n_jobs=1):
        self.datasets = datasets
        self.estimators = estimators
        self.param_grids = param_grids
        self.scoring = scoring
        self.cv = cv
        self.experiment_repetitions = experiment_repetitions
        self.random_state = random_state
        self.n_jobs = n_jobs

    def run(self, hide_warnings=True):
        """Runs the experimental procedure and calculates the cross validation
        scores for each dataset, pipeline and hyperparameters."""

        # Remove warnings
        if hide_warnings:
            filterwarnings('ignore')
        else:
            filterwarnings('default')

        # Initialize experiment parameters
        if not hasattr(self, 'datasets'):
            return
        datasets = check_datasets(self.datasets)
        self.datasets_names_ = [dataset_name for dataset_name, _ in datasets]
        self.param_grids_ = dict(zip([estimator_name for estimator_name, _ in self.estimators], self.param_grids))
        self.cv_ = check_cv(self.cv)
        self.cv_.shuffle = True
        self.random_states_ = check_random_states(self.random_state, self.experiment_repetitions)

        # Initialize progress bar
        progress_bar = ProgressBar(redirect_stdout=False, max_value=len(self.random_states_) * len(datasets) * len(self.estimators))
        iterations = 0

        # Create all possible combination of experimental configurations
        combinations = product(self.random_states_, datasets, self.estimators)

        # Run the experiment
        self.results_ = []
        for random_state, (dataset_name, (X, y)), (estimator_name, estimator) in combinations:
            self.cv_.random_state = random_state
            gscv = GridSearchCV(estimator, self.param_grids_[estimator_name], self.scoring, cv=self.cv_, refit=False)
            #gscv.fit(X, y)
            self.results_.append((dataset_name, gscv))
            iterations += 1
            progress_bar.update(iterations)

    def save(self, filename, pickle_datasets=False):
        """Pickles the experiment object."""
        if not pickle_datasets:
            if hasattr(self, 'datasets'):
                delattr(self, 'datasets')
        dump(self, open(filename, 'wb'))
