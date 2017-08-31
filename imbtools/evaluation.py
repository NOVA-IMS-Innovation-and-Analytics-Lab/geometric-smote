"""
This module contains classes to compare and evaluate
the performance of various oversampling algorithms.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from warnings import filterwarnings
from itertools import product
from os.path import join
from os import listdir
from re import match, sub
from pickle import dump, load
from .utils import check_datasets, check_random_states, check_models
from imblearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score
from scipy.stats import friedmanchisquare
from progressbar import ProgressBar
from .metrics import SCORERS
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd


def read_csv_dir(dirpath):
    "Reads a directory of csv files and returns a dictionary of dataset-name:(X,y) pairs."
    datasets = []
    csv_files = [csv_file for csv_file in listdir(dirpath) if match('^.+\.csv$', csv_file)]
    for csv_file in csv_files:
        dataset = pd.read_csv(join(dirpath, csv_file))
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        dataset_name = sub(".csv", "", csv_file)
        datasets.append((dataset_name, (X, y)))
    return datasets

def summarize_datasets(datasets):
    """Creates a summary of the datasets."""
    datasets = check_datasets(datasets)
    summary_columns = ["Dataset name",
                       "# of features",
                       "# of instances",
                       "# of minority instances",
                       "# of majority instances",
                       "Imbalance Ratio"]
    datasets_summary = pd.DataFrame({}, columns=summary_columns)
    for dataset_name, (X, y) in datasets:
        n_instances = ((y == 0).sum(), (y == 1).sum())
        dataset_summary = pd.DataFrame([[dataset_name,
                                         X.shape[1],
                                         y.size,
                                         n_instances[1],
                                         n_instances[0],
                                         round(n_instances[0] / n_instances[1], 2)]],
                                       columns=datasets_summary.columns)
        datasets_summary = datasets_summary.append(dataset_summary, ignore_index=True)
    datasets_summary[datasets_summary.columns[1:-1]] = datasets_summary[datasets_summary.columns[1:-1]].astype(int)
    return datasets_summary

def _calculate_stats(experiment):
    """Calculates stats for positive and negative scorers."""
    grouped_results = experiment.results_.groupby(experiment.results_.columns[:-1].tolist(), as_index=False)
    stats = grouped_results.agg({'CV score': [np.mean, np.std]})
    stats.columns = experiment.results_.columns.tolist()[:-1] + ['Mean CV score', 'Std CV score']
    return stats

def calculate_stats(experiment):
    """Calculates mean and standard deviation across experiments for every
    combination of datasets, classifiers, oversamplers and metrics."""
    stats = _calculate_stats(experiment)
    stats['Mean CV score'] = np.abs(stats['Mean CV score'])
    return stats

def calculate_optimal_stats(experiment):
    """Calculates the highest mean and standard deviation for every
    combination of classfiers and oversamplers across different
    hyperparameters' configurations."""

    # Classifiers names
    clfs_names = [clf_name for clf_name, *_ in experiment.classifiers]
    expanded_clfs_names = [clf_name for clf_name, _ in experiment.classifiers_]

    # Oversamplers names
    oversamplers_names = [oversampler_name for oversampler_name, *_ in experiment.oversamplers]
    expanded_oversamplers_names = [oversampler_name for oversampler_name, _ in experiment.oversamplers_]

    # Calculate stats table
    stats = _calculate_stats(experiment)
    optimal_stats = pd.DataFrame(columns=stats.columns)

    # Populate optimal stats table
    for dataset_name, clf_name, oversampler_name in product(experiment.datasets_names_, clfs_names, oversamplers_names):
        matched_clfs_names = [exp_clf_name for exp_clf_name in expanded_clfs_names if match(clf_name, exp_clf_name)]
        matched_oversamplers_names = [exp_oversampler_name for exp_oversampler_name in expanded_oversamplers_names if match(oversampler_name, exp_oversampler_name)]

        is_matched_clfs = np.isin(stats['Classifier'], matched_clfs_names)
        is_matched_oversamplers = np.isin(stats['Oversampler'], matched_oversamplers_names)
        is_matched_dataset = (stats['Dataset'] == dataset_name)

        matched_stats = stats[is_matched_clfs & is_matched_oversamplers & is_matched_dataset]
        optimal_matched_stats = matched_stats.groupby('Metric', as_index=False).agg({'Mean CV score': [max, lambda col: matched_stats['Std CV score'][np.argmax(col)]]})
        optimal_matched_stats.columns = stats.columns[-3:]
        optimal_matched_names = pd.DataFrame([[dataset_name, clf_name, oversampler_name]] * len(experiment.scoring), columns=stats.columns[:-3])
        optimal_matched_stats = pd.concat([optimal_matched_names, optimal_matched_stats], axis=1)
        optimal_stats = optimal_stats.append(optimal_matched_stats, ignore_index=True)

    optimal_stats['Mean CV score'] = np.abs(optimal_stats['Mean CV score'])
    return optimal_stats

def calculate_optimal_stats_wide(experiment):
    """Calculates in wide format the highest mean and standard
    deviation for every combination of classfiers and oversamplers
    across different hyperparameters' configurations."""
    optimal_stats = calculate_optimal_stats(experiment)

    # Calculate wide format of mean cv
    optimal_mean_wide = optimal_stats.pivot_table(index=['Dataset', 'Classifier', 'Metric'], columns=['Oversampler'], values='Mean CV score').reset_index()
    optimal_mean_wide.columns.rename(None, inplace=True)

    # Calculate wide format of std cv
    optimal_std_wide = optimal_stats.pivot_table(index=['Dataset', 'Classifier', 'Metric'], columns=['Oversampler'], values='Std CV score').reset_index()
    optimal_std_wide.columns.rename(None, inplace=True)

    # Polpulate wide format of optimal stats
    oversamplers_names = [oversampler_name for oversampler_name, *_ in experiment.oversamplers]
    optimal_stats_wide = pd.DataFrame(columns=oversamplers_names)
    for oversampler_name in oversamplers_names:
        optimal_stats_wide[oversampler_name] = list(zip(optimal_mean_wide[oversampler_name], optimal_std_wide[oversampler_name]))
    return pd.concat([optimal_mean_wide[['Dataset', 'Classifier', 'Metric']], optimal_stats_wide], axis=1)

def calculate_ranking(experiment):
    """Calculates the ranking of oversamplers."""
    optimal_stats_wide = calculate_optimal_stats_wide(experiment)
    ranking = optimal_stats_wide.apply(lambda row: len(row[3:]) - row[3:].argsort().argsort(), axis=1)
    return pd.concat([optimal_stats_wide.iloc[:, :3], ranking], axis=1)

def calculate_mean_ranking(experiment):
    """Calculates the ranking of oversamplers."""
    ranking = calculate_ranking(experiment)
    return ranking.groupby(['Classifier', 'Metric'], as_index=False).mean()

def calculate_friedman_test(experiment):
    """Calculates the friedman test across datasets for every
    combination of classifiers and metrics."""
    if len(experiment.oversamplers_) < 3:
        raise ValueError('Friedman test can not be applied. More than two oversampling methods are needed.')
    ranking = calculate_ranking(experiment)
    extract_pvalue = lambda df: friedmanchisquare(*df.iloc[:, 3:].transpose().values.tolist()).pvalue
    return ranking.groupby(['Classifier', 'Metric']).apply(extract_pvalue).reset_index().rename(columns={0: 'p-value'})

def load_experiment(filename):
    """Loads a saved experiment object."""
    loaded_obj = load(open(filename, 'rb'))
    if not isinstance(loaded_obj, BinaryExperiment):
        raise TypeError("File {} is not a BinaryExperiment instance.")
    return loaded_obj


class BinaryExperiment:
    """Class for comparison of oversampling algorithms performance
    on imbalanced binary classification problems.

    Parameters
    ----------
    datasets : list of (X, y) tuples or dictionary of dataset-name:(X,y) pairs
        The list of (X, y) pairs is a list of tuples of input data and
        target values, The dictionary extends the list by adding the datasets names
        as a key.
    classifiers : list of classifiers
        A list of classifiers.
    oversamplers : list of oversamplers
        A list of oversampling methods.
    metrics : list of string scorers, (default=[᾽roc_auc᾽, ᾽f1᾽])
        A list of classification metrics.
    n_splits : int, (default=3)
        The number of cross validation stages.
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
                 classifiers,
                 oversamplers,
                 scoring=['roc_auc', 'f1'],
                 n_splits=3,
                 experiment_repetitions=5,
                 random_state=None,
                 n_jobs=1):
        self.datasets = datasets
        self.classifiers = classifiers
        self.oversamplers = oversamplers
        self.scoring = scoring
        self.n_splits = n_splits
        self.experiment_repetitions = experiment_repetitions
        self.random_state = random_state
        self.n_jobs = n_jobs

    def run(self, hide_warnings=True):
        """Runs the experimental procedure and calculates the cross validation
        scores for each classifier, oversampling method, datasets and metric."""

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
        self.random_states_ = check_random_states(self.random_state, self.experiment_repetitions)
        self.classifiers_ = check_models(self.classifiers, "classifier")
        self.oversamplers_ = check_models(self.oversamplers, "oversampler")

        # Initialize progress bar
        progress_bar = ProgressBar(redirect_stdout=False, max_value=len(self.random_states_) * len(datasets) * len(self.classifiers_) * len(self.oversamplers_))
        iterations = 0

        # Populate results dataframe
        results_columns = ['Dataset', 'Classifier', 'Oversampler', 'Metric', 'CV score']
        self.results_ = pd.DataFrame(columns=results_columns)
        for random_state in self.random_states_:
            cv = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
            for dataset_name, (X, y) in datasets:
                for clf_name, clf, in self.classifiers_:
                    if 'random_state' in clf.get_params().keys():
                        clf.set_params(random_state=random_state)
                    for oversampler_name, oversampler in self.oversamplers_:
                        if oversampler is not None:
                            oversampler.set_params(random_state=random_state)
                            clf = Pipeline([(oversampler_name, oversampler), (clf_name, clf)])
                        cv_output = cross_validate(clf, X, y, cv=cv, scoring=self.scoring, n_jobs=self.n_jobs)
                        iterations += 1
                        progress_bar.update(iterations)
                        for scorer in self.scoring:
                            cv_score = cv_output["test_" + scorer].mean()
                            result_list = [dataset_name, clf_name, oversampler_name, scorer, cv_score]
                            result = pd.DataFrame([result_list], columns=results_columns)
                            self.results_ = self.results_.append(result, ignore_index=True)

    def save(self, filename, pickle_datasets=False):
        """Pickles the experiment object."""
        if not pickle_datasets:
            if hasattr(self, 'datasets'):
                delattr(self, 'datasets')
        dump(self, open(filename, 'wb'))
