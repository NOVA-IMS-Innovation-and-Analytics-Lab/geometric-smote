"""
The :mod:`sklearnext.tools.imbalanced_analysis` module
contains functions to evaluate the results of model search.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from collections import Counter
from os.path import join
from os import listdir
from re import match, sub

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold

from ..utils import check_datasets, check_oversamplers_classifiers
from ..utils.estimators import _ParametrizedEstimatorsMixin
from ..metrics import SCORERS
from ..model_selection import ModelSearchCV

GROUP_KEYS = ['Dataset', 'Oversampler', 'Classifier', 'params']


def read_csv_dir(dirpath):
    "Reads a directory of csv files and returns a dictionary of dataset-name:(X,y) pairs."
    
    # Define empty datasets list to return
    datasets = []

    # Read csv filenames
    csv_files = [csv_file for csv_file in listdir(dirpath) if match('^.+\.csv$', csv_file)]
    
    # Populate datasets list
    for csv_file in csv_files:
        dataset = pd.read_csv(join(dirpath, csv_file))
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        dataset_name = sub(".csv", "", csv_file)
        datasets.append((dataset_name, (X, y)))

    return datasets


def summarize_binary_datasets(datasets):
    """Creates a summary of the binary class
    imbalanced datasets."""

    # Check datasets format
    datasets = check_datasets(datasets)
    
    # Define summary table columns
    summary_columns = ["Dataset name",
                       "Features",
                       "Instances",
                       "Minority instances",
                       "Majority instances",
                       "Imbalance Ratio"]
    
    # Define empty summary table
    datasets_summary = pd.DataFrame({}, columns=summary_columns)
    
    # Populate summary table
    for dataset_name, (X, y) in datasets:
        n_instances = Counter(y).values()
        n_minority_instances, n_majority_instances = min(n_instances), max(n_instances)
        dataset_summary = pd.DataFrame([[dataset_name,
                                         X.shape[1],
                                         len(X),
                                         n_minority_instances,
                                         n_majority_instances,
                                         round(n_majority_instances / n_minority_instances, 2)]],
                                       columns=summary_columns)
        datasets_summary = datasets_summary.append(dataset_summary, ignore_index=True)
    
    # Cast to integer columns
    datasets_summary[datasets_summary.columns[1:-1]] = datasets_summary[datasets_summary.columns[1:-1]].astype(int)

    return datasets_summary.sort_values('Imbalance Ratio').reset_index(drop=True)


def _define_binary_experiment_parameters(model_search_cv):
    """Define the binary experiment parameters."""
    
    # Get scoring attribute
    scoring = model_search_cv.scoring

    # Define scoring columns
    if isinstance(scoring, list):
        scoring_cols = ['mean_test_%s' % scorer for scorer in scoring]
    else:
        scoring_cols = ['mean_test_score']

    # Get estimator type attribute
    estimator_type = model_search_cv.estimator._estimator_type

    return scoring, scoring_cols, estimator_type


def _set_verbose_attributes(ind, dataset_name, datasets):
    """Set attributes used when experimental procedure is verbose."""
    for attribute, value in zip(['ind', 'dataset_name', 'n_datasets'], [ind, dataset_name, len(datasets)]):
        setattr(_ParametrizedEstimatorsMixin, attribute, value)


def _calculate_results(model_search_cv, datasets, scoring_cols, verbose):
    """Calculates the results of binary imbalanced experiments."""

    # Define empty results table
    results = pd.DataFrame()

    # Populate results table
    for ind, (dataset_name, (X, y)) in enumerate(datasets):

        # set verbose attributes
        if verbose:
            _set_verbose_attributes(ind, dataset_name, datasets)
        
        # Fit model search
        model_search_cv.fit(X, y)
        
        # Get results
        result = pd.DataFrame(model_search_cv.cv_results_).loc[:, ['models', 'params'] + scoring_cols]
        
        # Append dataset name column
        result = result.assign(Dataset=dataset_name)

        # Append result
        results = results.append(result)

    return results


def _calculate_aggregated_results(results, scoring_cols):
    """Calculates the aggregated results across datasets of binary
    imbalanced experiments."""

    # Copy results
    aggregated_results = results.copy()
    
    # Parse oversamplers and classifiers
    aggregated_results.loc[:, 'models'] = aggregated_results.loc[:, 'models'].apply(lambda model: sub('_[0-9]*$', '', model).split('|'))
    aggregated_results[['Oversampler', 'Classifier']] = pd.DataFrame(aggregated_results.models.values.tolist())
    
    # Cast parameters to string
    aggregated_results['params'] = aggregated_results.loc[:, 'params'].apply(str).drop(columns='models')

    # Calculate aggregated mean and standard deviation
    scoring_mapping = {scorer_name: [np.mean, np.std] for scorer_name in scoring_cols}
    aggregated_results = aggregated_results.groupby(GROUP_KEYS).agg(scoring_mapping)

    return aggregated_results


def _calculate_optimal_results(aggregated_results, datasets_names, oversamplers_names, classifiers_names, scoring_cols):
    """"Calculate optimal results across hyperparameters for any combination of
    datasets, overamplers, classifiers and metrics."""

    # Select mean scores
    optimal_results = aggregated_results[[(score, 'mean') for score in scoring_cols]].reset_index()
    
    # Flatten columns
    optimal_results.columns = optimal_results.columns.get_level_values(0)
    
    # Calculate maximum score per gorup key
    agg_measures = {score: max for score in scoring_cols}
    optimal_results = optimal_results.groupby(GROUP_KEYS[:-1]).agg(agg_measures).reset_index()
    
    # Format as long table
    optimal_results = optimal_results.melt(id_vars=GROUP_KEYS[:-1],
                                           value_vars=scoring_cols,
                                           var_name='Metric',
                                           value_name='Score')
    
    # Cast to categorical columns
    optimal_results_cols = GROUP_KEYS[:-1] + ['Metric']
    names = [datasets_names, oversamplers_names, classifiers_names, scoring_cols]
    for col, name in zip(optimal_results_cols, names):
        optimal_results[col] = pd.Categorical(optimal_results[col], name)
    
    # Sort values
    optimal_results = optimal_results.sort_values(optimal_results_cols).reset_index(drop=True)
    
    return optimal_results


def _calculate_wide_optimal_results(optimal_results, scoring, estimator_type):
    """Calculate optimal results in wide format."""

    # Format as wide table
    wide_optimal_results = optimal_results.pivot_table(index=['Dataset', 'Classifier', 'Metric'],
                                                       columns=['Oversampler'],
                                                       values='Score')
    wide_optimal_results.columns = wide_optimal_results.columns.tolist()
    wide_optimal_results.reset_index(inplace=True)
    
    # Transform metric column
    if isinstance(scoring, list):
        wide_optimal_results['Metric'] = wide_optimal_results['Metric'].replace('mean_test_', '', regex=True)
    elif isinstance(scoring, str):
        wide_optimal_results['Metric'] = scoring
    else:
        wide_optimal_results['Metric'] = 'accuracy' if estimator_type == 'classifier' else 'r2'

    # Cast to categorical
    wide_optimal_results['Metric'] = pd.Categorical(wide_optimal_results['Metric'],
                                                    categories=scoring if isinstance(scoring, list) else None)
    return wide_optimal_results


def _return_row_ranking(row, sign):
    """Returns the ranking of mean cv scores for
    a row of an array. In case of tie, each value
    is replaced with its group average."""

    # Calculate ranking
    ranking = (sign * row).argsort().argsort().astype(float)
    
    # Break the tie
    groups = np.unique(row, return_inverse=True)[1]
    for group_label in np.unique(groups):
        indices = (groups == group_label)
        ranking[indices] = ranking[indices].mean()

    return ranking.size - ranking


def _calculate_ranking_results(wide_optimal_results):
    """Calculate the ranking of oversamplers for
    any combination of datasets, classifiers and
    metrics."""
    ranking_results = wide_optimal_results.apply(lambda row: _return_row_ranking(row[3:], SCORERS[row[2]]._sign), axis=1)
    ranking_results = pd.concat([wide_optimal_results.iloc[:, :3], ranking_results], axis=1)
    return ranking_results


def _calculate_friedman_test_results(ranking_results, alpha=0.05):
    """Calculate the Friedman test across datasets for every
    combination of classifiers and metrics."""

    # Raise an error when less then three oversamplers are used
    if len(ranking_results.columns) < 6:
        raise ValueError('Friedman test can not be applied. More than two oversampling methods are needed.')

    # DEfine function that calculate p-value
    extract_pvalue = lambda df: friedmanchisquare(*df.iloc[:, 3:].transpose().values.tolist()).pvalue
    
    # Calculate p-values
    friedman_test_results = ranking_results.groupby(['Classifier', 'Metric']).apply(extract_pvalue).reset_index().rename(
        columns={0: 'p-value'})
    
    # Compare p-values to significance level
    friedman_test_results['Significance'] = friedman_test_results['p-value'] < alpha

    return friedman_test_results


def _calculate_adjusted_pvalues_results(wide_optimal_results, control_oversampler):
    """Use the Holm's method to adjust the p-values of a paired difference
    t-test for every combination of classifiers and metrics using a control
    oversampler."""

    # Get the oversamplers name
    oversamplers_names = wide_optimal_results.columns[3:].tolist()
    
    # Use the last if no control oversampler is provided
    if control_oversampler is None:
        control_oversampler = oversamplers_names[-1]
    oversamplers_names.remove(control_oversampler)
    
    # Define empty p-values table
    pvalues = pd.DataFrame()

    # Populate p-values table
    for name in oversamplers_names:
        pvalues_pair = wide_optimal_results.groupby(['Classifier', 'Metric'])[[name, control_oversampler]].apply(
            lambda df: ttest_rel(df[name], df[control_oversampler])[1])
        pvalues_pair = pd.DataFrame(pvalues_pair, columns=[name])
        pvalues = pd.concat([pvalues, pvalues_pair], axis=1)
    corrected_pvalues = pd.DataFrame(pvalues.apply(
        lambda col: multipletests(col, method='holm')[1], axis=1).values.tolist(), columns=oversamplers_names)
    corrected_pvalues = corrected_pvalues.set_index(pvalues.index).reset_index()
    
    return corrected_pvalues


def _format_metrics(results):
    """Pretty format the metric names."""
    results['Metric'] = results['Metric'].replace('_', ' ', regex=True).apply(lambda metric: metric.upper())
    return results


def evaluate_binary_imbalanced_experiments(datasets, oversamplers, classifiers, scoring=None, alpha=0.05,
                                           control_oversampler=None, n_splits=3, n_runs=3, random_state=None,
                                           verbose=True, scheduler=None, n_jobs=-1, cache_cv=True):

    # Extract estimators and parameter grids
    estimators, param_grids = check_oversamplers_classifiers(oversamplers, classifiers, n_runs, random_state).values()
    mscv = ModelSearchCV(estimators,
                         param_grids,
                         scoring=scoring,
                         iid=True,
                         refit=False,
                         cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
                         error_score='raise',
                         return_train_score=False,
                         scheduler=scheduler,
                         n_jobs=n_jobs,
                         cache_cv=cache_cv,
                         verbose=verbose)

    # Define experiment parameters
    datasets_names, _ = zip(*datasets)
    oversamplers_names, _ = zip(*oversamplers)
    classifiers_names, _ = zip(*classifiers)
    scoring, scoring_cols, estimator_type = _define_binary_experiment_parameters(mscv)

    # Results
    results = _calculate_results(mscv, datasets, scoring_cols, verbose)
    aggregated_results = _calculate_aggregated_results(results, scoring_cols)
    optimal_results = _calculate_optimal_results(aggregated_results, datasets_names, oversamplers_names,
                                                 classifiers_names, scoring_cols)
    wide_optimal_results = _calculate_wide_optimal_results(optimal_results, scoring, estimator_type)
    ranking_results = _calculate_ranking_results(wide_optimal_results)
    mean_ranking_results = ranking_results.groupby(['Classifier', 'Metric'], as_index=False).mean()
    friedman_test_results = _calculate_friedman_test_results(ranking_results, alpha)
    adjusted_pvalues_results = _calculate_adjusted_pvalues_results(wide_optimal_results, control_oversampler)

    return {'aggregated': aggregated_results,
            'optimal': optimal_results,
            'wide_optimal': _format_metrics(wide_optimal_results),
            'ranking': _format_metrics(ranking_results),
            'mean_ranking': _format_metrics(mean_ranking_results),
            'friedman_test': _format_metrics(friedman_test_results),
            'adjusted_pvalues': -_format_metrics(adjusted_pvalues_results)}
