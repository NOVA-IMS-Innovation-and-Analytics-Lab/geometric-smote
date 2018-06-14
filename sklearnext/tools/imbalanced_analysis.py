"""
The :mod:`sklearnext.tools.imbalanced_analysis` module
contains functions to evaluate the results of model search.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from os.path import join
from os import listdir
from re import match, sub
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from sklearn.model_selection import StratifiedKFold
from ..utils import check_datasets, check_oversamplers_classifiers
from ..metrics import SCORERS
from ..model_selection import ModelSearchCV


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


def _return_row_ranking(row, sign):
    """Returns the ranking of mean cv scores for
    a row of an array. In case of tie, each value
    is replaced with its group average."""
    ranking = (sign * row).argsort().argsort().astype(float)
    groups = np.unique(row, return_inverse=True)[1]
    for group_label in np.unique(groups):
        indices = (groups == group_label)
        ranking[indices] = ranking[indices].mean()
    return ranking.size - ranking


def calculate_friedman_test_results(ranking_results, alpha=0.05):
    """Calculates the friedman test across datasets for every
    combination of classifiers and metrics."""
    if len(ranking_results.columns) < 6:
        raise ValueError('Friedman test can not be applied. More than two oversampling methods are needed.')
    extract_pvalue = lambda df: friedmanchisquare(*df.iloc[:, 3:].transpose().values.tolist()).pvalue
    friedman_test_results = ranking_results.groupby(['Classifier', 'Metric']).apply(extract_pvalue).reset_index().rename(
        columns={0: 'p-value'})
    friedman_test_results['Significance'] = friedman_test_results['p-value'] < alpha
    return friedman_test_results


def evaluate_imbalanced_experiments(datasets, oversamplers, classifiers, scoring=None, alpha=0.05,
                                    n_splits=3, n_runs=3, random_state=None, n_jobs=-1):
    estimators, param_grids = check_oversamplers_classifiers(oversamplers, classifiers, n_runs, random_state).values()
    mscv = ModelSearchCV(estimators,
                         param_grids,
                         scoring=scoring,
                         iid=True,
                         refit=False,
                         cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                         error_score='raise',
                         return_train_score=False,
                         scheduler=None,
                         n_jobs=n_jobs,
                         cache_cv=True)
    if isinstance(mscv.scoring, list):
        scoring_cols = ['mean_test_%s' % scorer for scorer in mscv.scoring]
    else:
        scoring_cols = ['mean_test_score']
    results = pd.DataFrame()
    for dataset_name, (X, y) in datasets:
        mscv.fit(X, y)
        result = pd.DataFrame(mscv.cv_results_).loc[:, ['models', 'params'] + scoring_cols]
        result = result.assign(Dataset=dataset_name)
        results = results.append(result)

    # Aggregate results across experiments
    results.loc[:, 'models'] = results.loc[:, 'models'].apply(lambda model: sub('_[0-9]*$', '', model).split('|'))
    results[['Oversampler', 'Classifier']] = pd.DataFrame(results.models.values.tolist())
    results['params'] = results.loc[:, 'params'].apply(str).drop(columns='models')

    scoring_mapping = {scorer_name: [np.mean, np.std] for scorer_name in scoring_cols}
    group_keys = ['Dataset', 'Oversampler', 'Classifier', 'params']
    aggregated_results = results.groupby(group_keys).agg(scoring_mapping)

    # Calculate optimal results across hyperparameters
    optimal_results = aggregated_results[[(score, 'mean') for score in scoring_cols]].reset_index()
    optimal_results.columns = optimal_results.columns.get_level_values(0)
    agg_measures = {score:max for score in scoring_cols}
    optimal_results = optimal_results.groupby(group_keys[:-1]).agg(agg_measures).reset_index()
    optimal_results = optimal_results.melt(id_vars=['Dataset', 'Oversampler', 'Classifier'],
                                           value_vars=scoring_cols,
                                           var_name='Metric',
                                           value_name='Score')

    # Calculate optimal results across hyperparameters in wide format
    wide_optimal_results = optimal_results.pivot_table(index=['Dataset', 'Classifier', 'Metric'],
                                                       columns=['Oversampler'],
                                                       values='Score').reset_index()
    wide_optimal_results.columns.rename(None, inplace=True)
    if isinstance(mscv.scoring, list):
        wide_optimal_results['Metric'].replace('mean_test_', '', regex=True, inplace=True)
    elif isinstance(mscv.scoring, list):
        wide_optimal_results['Metric'] = mscv.scoring
    else:
        wide_optimal_results['Metric'] = 'accuracy' if mscv.estimator._estimator_type == 'classifier' else 'r2'

    # Calculate ranking
    ranking_results = wide_optimal_results.apply(lambda row: _return_row_ranking(row[3:], SCORERS[row[2]]._sign), axis=1)
    ranking_results = pd.concat([wide_optimal_results.iloc[:, :3], ranking_results], axis=1)

    # Calculate mean ranking
    mean_ranking_results = ranking_results.groupby(['Classifier', 'Metric'], as_index=False).mean()

    # Calculate Friedman test results
    friedman_test_results = calculate_friedman_test_results(ranking_results, alpha)

    return {'aggregated_results': aggregated_results,
            'optimal_results': optimal_results,
            'wide_optimal_results': wide_optimal_results,
            'ranking_results': ranking_results,
            'mean_ranking_results': mean_ranking_results,
            'friedman_test_results': friedman_test_results}
