"""
This module contains the class to compare pipelines
with various hyperparameters.
"""

from warnings import filterwarnings
from pickle import dump, load
from progressbar import ProgressBar
import pandas as pd
from sklearn.model_selection import GridSearchCV
from .utils import check_pipelines, check_param_grids


DEFAULT_SCORING = ['mean_squared_error']

def load_pipelines(filename):
    """Loads a saved Pipelines object."""
    loaded_obj = load(open(filename, 'rb'))
    if not isinstance(loaded_obj, Pipelines):
        raise TypeError("File {} is not a Pipelines instance.")
    return loaded_obj

class Pipelines:
    """Class for fitting various pipelines of transformations/estimators.

    Parameters
    ----------
    pipelines : list of (pipeline name, steps, param_grid) tuples
        The steps parameter is the steps parameter of the Pipeline class.
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
    refit : boolean, or string, default=True
        Refit an estimator using the best found parameters on the whole
        dataset. For multiple metric evaluation, this needs to be a string
        denoting the scorer is used to find the best parameters for refitting
        the estimator at the end.
    n_jobs : int, (default=-1)
        The number of CPUs to use to do the computation. -1 means ‘all CPUs’.
    """

    def __init__(self,
                 pipelines,
                 scoring=None,
                 cv=None,
                 refit=None,
                 n_jobs=-1):
        self.pipelines = pipelines
        self.scoring = scoring
        self.cv = cv
        self.refit = refit
        self.n_jobs = n_jobs

    def fit(self, X, y, hide_warnings=True):
        """Fits all the the pipelines."""

        # Remove warnings
        if hide_warnings:
            filterwarnings('ignore')
        else:
            filterwarnings('default')

        # Check parameters
        self.names_ = [name for name, _ in self.pipelines]
        pipelines = [estimator for _, estimator in self.pipelines]
        self.pipelines_ = check_pipelines(pipelines)
        self.param_grids_ = check_param_grids(pipelines)
        self.scoring_ = DEFAULT_SCORING if self.scoring is None else self.scoring

        # Initialize progress bar
        progress_bar = ProgressBar(redirect_stdout=False, max_value=len(self.pipelines_))
        iterations = 0

        # Fit the pipelines
        self.results_ = []
        for estimator, param_grid in zip(self.pipelines_, self.param_grids_):
            gscv = GridSearchCV(estimator,
                                list(param_grid) if param_grid else param_grid,
                                self.scoring_,
                                cv=self.cv,
                                refit=False,
                                return_train_score=False,
                                n_jobs=self.n_jobs)
            gscv.fit(X, y)
            self.results_.append(gscv)
            iterations += 1
            progress_bar.update(iterations)

        # Fit the best pipeline
        if self.refit is not None:
            results = self.extract_results()
            if isinstance(self.refit, str):
                best_ind = results[self.refit].argmax()
            elif self.refit is True:
                best_ind = results[self.scoring_[0]].argmax()
            best_pipeline_name = results.loc[best_ind, 'Pipeline']
            self.best_pipeline_ = self.pipelines_[self.names_.index(best_pipeline_name)]
            best_params = results.loc[best_ind, 'Parameters']
            self.best_pipeline_.set_params(**best_params)
            self.best_pipeline_.fit(X, y)

    def extract_results(self):
        """Extracts the results of a pipelines
        comparison experiment in a pandas dataframe."""
        scores = ['mean_test_' + scorer for scorer in self.scoring_]
        cols = ['params'] + scores
        results = pd.DataFrame()
        for name, gscv in zip(self.names_, self.results_):
            cv_results = pd.DataFrame({k:v for k, v in gscv.cv_results_.items() if k in cols})
            cv_results['Pipeline'] = [name] * len(cv_results)
            results = results.append(cv_results)
        results = results[['Pipeline'] + cols]
        renamed_cols = [col.replace('mean_test_', ' ')[1:] for col in scores]
        renamed_cols = ['Parameters'] + renamed_cols
        columns = {k:v for k, v in zip(cols, renamed_cols)}
        results = results.rename(columns=columns).reset_index(drop=True)
        return results

    def save(self, filename):
        """Pickles the Pipelines object."""
        dump(self, open(filename, 'wb'))
