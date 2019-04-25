"""
The :mod:`sklearnext.tools.imbalanced_analysis` module
contains functions and classes to evaluate the results of model search.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from collections import Counter
from re import sub
from os.path import join
from pickle import dump

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold

from ..utils import check_datasets, check_oversamplers_classifiers
from ..metrics import SCORERS
from ..model_selection import ModelSearchCV

GROUP_KEYS = ['Dataset', 'Oversampler', 'Classifier', 'params']


class BinaryExperiment:
    """Define an experiment for binary classification on imbalanced datasets."""

    def __init__(self, name, datasets, oversamplers, classifiers, scoring, n_splits, n_runs, random_state):
        self.name = name
        self.datasets = datasets
        self.oversamplers = oversamplers
        self.classifiers = classifiers
        self.scoring = scoring
        self.n_splits = n_splits
        self.n_runs = n_runs
        self.random_state = random_state

    def summarize_datasets(self):
        """Creates a summary of the binary class
        imbalanced datasets."""

        # Check datasets format
        datasets = check_datasets(self.datasets)
    
        # Define summary table columns
        summary_columns = ["Dataset name", "Features", "Instances", "Minority instances", "Majority instances", "Imbalance Ratio"]
    
        # Define empty summary table
        datasets_summary = pd.DataFrame({}, columns=summary_columns)
    
        # Populate summary table
        for dataset_name, (X, y) in datasets:
            n_instances = Counter(y).values()
            n_minority_instances, n_majority_instances = min(n_instances), max(n_instances)
            values = [dataset_name, X.shape[1], len(X), n_minority_instances, n_majority_instances, round(n_majority_instances / n_minority_instances, 2)]
            dataset_summary = pd.DataFrame([values], columns=summary_columns)
            datasets_summary = datasets_summary.append(dataset_summary, ignore_index=True)
    
        # Cast to integer columns
        datasets_summary[datasets_summary.columns[1:-1]] = datasets_summary[datasets_summary.columns[1:-1]].astype(int)

        self.datasets_summary_ = datasets_summary.sort_values('Imbalance Ratio').reset_index(drop=True)

    def _initialize(self, n_jobs, verbose):
        """Initialize experiment's parameters."""

        # Scoring columns
        if isinstance(self.scoring, list):
            self.scoring_cols_ = ['mean_test_%s' % scorer for scorer in self.scoring]
        else:
            self.scoring_cols_ = ['mean_test_score']

        # Datasets, oversamplers and classifiers
        self.datasets_names_, _ = zip(*self.datasets)
        self.oversamplers_names_, *_ = zip(*self.oversamplers)
        self.classifiers_names_, *_ = zip(*self.classifiers)

        # Extract estimators and parameter grids
        self.estimators_, self.param_grids_ = check_oversamplers_classifiers(self.oversamplers, self.classifiers, self.n_runs, self.random_state).values()

        # Create model search cv
        self.mscv_ = ModelSearchCV(self.estimators_, self.param_grids_, scoring=self.scoring, iid=True, refit=False, 
                                   cv=StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state),
                                   return_train_score=False, n_jobs=n_jobs, verbose=verbose)
        
    def _check_results(self):
        """Check if results have generated."""
        if not hasattr(self, 'results_'):
            raise AttributeError('No results were found. Run experiment first.')
        
    def run(self, n_jobs=-1, verbose=0):
        """Run experiment."""

        self._initialize(n_jobs, verbose)

        # Define empty results table
        results = pd.DataFrame()

        # Populate results table
        for dataset_name, (X, y) in self.datasets:
        
            # Fit model search
            self.mscv_.fit(X, y)
        
            # Get results
            result = pd.DataFrame(self.mscv_.cv_results_).loc[:, ['models', 'params'] + self.scoring_cols_]
        
            # Append dataset name column
            result = result.assign(Dataset=dataset_name)

            # Append result
            results = results.append(result)

        # Calculate aggregated experimental results across runs
        results.loc[:, 'models'] = results.loc[:, 'models'].apply(lambda model: sub('_[0-9]*$', '', model).split('|'))
        results[['Oversampler', 'Classifier']] = pd.DataFrame(results.models.values.tolist())
    
        # Cast parameters to string
        results['params'] = results.loc[:, 'params'].apply(str).drop(columns='models')

        # Calculate aggregated mean and standard deviation
        scoring_mapping = {scorer_name: [np.mean, np.std] for scorer_name in self.scoring_cols_}
        self.results_ = results.groupby(GROUP_KEYS).agg(scoring_mapping)

        return self
    
    def calculate_optimal_results(self):
        """"Calculate optimal results across hyperparameters for any combination of datasets, overamplers, classifiers and metrics."""

        # Checks
        self._check_results()

        # Select mean scores
        optimal_results = self.results_[[(score, 'mean') for score in self.scoring_cols_]].reset_index()
    
        # Flatten columns
        optimal_results.columns = optimal_results.columns.get_level_values(0)
    
        # Calculate maximum score per gorup key
        agg_measures = {score: max for score in self.scoring_cols_}
        optimal_results = optimal_results.groupby(GROUP_KEYS[:-1]).agg(agg_measures).reset_index()
    
        # Format as long table
        optimal_results = optimal_results.melt(id_vars=GROUP_KEYS[:-1],
                                               value_vars=self.scoring_cols_,
                                               var_name='Metric',
                                               value_name='Score')
    
        # Cast to categorical columns
        optimal_results_cols = GROUP_KEYS[:-1] + ['Metric']
        names = [self.datasets_names_, self.oversamplers_names_, self.classifiers_names_, self.scoring_cols_]
        for col, name in zip(optimal_results_cols, names):
            optimal_results[col] = pd.Categorical(optimal_results[col], name)
    
        # Sort values
        self.optimal_results_ = optimal_results.sort_values(optimal_results_cols).reset_index(drop=True)

    def calculate_wide_optimal_results(self):
        """Calculate optimal results in wide format."""

        # Checks
        self._check_results()
        if not hasattr(self, 'optimal_results_'):
            self.calculate_optimal_results()

        # Format as wide table
        wide_optimal_results = self.optimal_results_.pivot_table(index=['Dataset', 'Classifier', 'Metric'],
                                                                 columns=['Oversampler'],
                                                                 values='Score')
        wide_optimal_results.columns = wide_optimal_results.columns.tolist()
        wide_optimal_results.reset_index(inplace=True)
    
        # Transform metric column
        if isinstance(self.scoring, list):
            wide_optimal_results['Metric'] = wide_optimal_results['Metric'].replace('mean_test_', '', regex=True)
        elif isinstance(self.scoring, str):
            wide_optimal_results['Metric'] = self.scoring
        else:
            wide_optimal_results['Metric'] = 'accuracy' if self.mscv_.estimator._estimator_type == 'classifier' else 'r2'

        # Cast column
        wide_optimal_results['Metric'] = pd.Categorical(wide_optimal_results['Metric'],
                                                        categories=self.scoring if isinstance(self.scoring, list) else None)

        self.wide_optimal_results_ = wide_optimal_results

    @staticmethod
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

    def calculate_ranking_results(self):
        """Calculate the ranking of oversamplers for
        any combination of datasets, classifiers and
        metrics."""

        # Checks
        self._check_results()
        if not hasattr(self, 'wide_optimal_results_'):
            self.calculate_wide_optimal_results()

        # Calculate ranking results
        ranking_results = self.wide_optimal_results_.apply(lambda row: self._return_row_ranking(row[3:], SCORERS[row[2].replace(' ', '_').lower()]._sign), axis=1)
        self.ranking_results_ = pd.concat([self.wide_optimal_results_.iloc[:, :3], ranking_results], axis=1)
    
    def calculate_mean_std_ranking_results(self):
        """Calculate the mean and std ranking of oversamplers 
        across datasets for any combination of classifiers 
        and metrics."""

        # Checks
        self._check_results()
        if not hasattr(self, 'ranking_results_'):
            self.calculate_ranking_results()

        self.mean_ranking_results_ = self.ranking_results_.groupby(['Classifier', 'Metric']).mean().reset_index()
        self.std_ranking_results_ = self.ranking_results_.groupby(['Classifier', 'Metric']).std().reset_index()

    def calculate_mean_std_scores(self):
        """Calculate mean and std scores across datasets."""

        # Checks
        self._check_results()
        if not hasattr(self, 'wide_optimal_results_'):
            self.calculate_wide_optimal_results()

        # Calculate mean and std score
        self.mean_scores_ = self.wide_optimal_results_.groupby(['Classifier', 'Metric']).mean().reset_index()
        self.std_scores_ = self.wide_optimal_results_.groupby(['Classifier', 'Metric']).std().reset_index()

    def calculate_perc_diff_mean_std_scores(self, oversamplers=None):
        """Calculate percentage difference of mean scores"""

        # Checks
        self._check_results()
        if not hasattr(self, 'mean_scores_'):
            self.calculate_wide_optimal_results()
        
        # Extract oversamplers
        control, test = oversamplers if oversamplers is not None else self.mean_scores_.columns[-2:]

        # Calculate percentage difference
        scores = self.wide_optimal_results_[self.wide_optimal_results_[basic] > 0]
        perc_diff_scores = pd.DataFrame((100 * (scores[test] - scores[control]) / scores[control]), columns=['Difference'])
        perc_diff_scores = pd.concat([scores.iloc[:, :3], perc_diff_scores], axis=1)

        # Calulate mean and std percentage difference
        self.mean_perc_diff_scores_ = perc_diff_scores.groupby(['Classifier', 'Metric']).mean().reset_index()
        self.std_perc_diff_scores_ = perc_diff_scores.groupby(['Classifier', 'Metric']).std().reset_index()

    def calculate_friedman_test_results(self, alpha=0.05):
        """Calculate the Friedman test across datasets for every
        combination of classifiers and metrics."""

        # Checks
        self._check_results()
        if not hasattr(self, 'ranking_results_'):
            self.calculate_ranking_results()

        # Raise an error when less then three oversamplers are used
        if len(self.ranking_results_.columns) < 6:
            raise ValueError('Friedman test can not be applied. More than two oversampling methods are needed.')

        # Define function that calculate p-value
        extract_pvalue = lambda df: friedmanchisquare(*df.iloc[:, 3:].transpose().values.tolist()).pvalue
    
        # Calculate p-values
        friedman_test_results = self.ranking_results_.groupby(['Classifier', 'Metric']).apply(extract_pvalue).reset_index().rename(
            columns={0: 'p-value'})
    
        # Compare p-values to significance level
        friedman_test_results['Significance'] = friedman_test_results['p-value'] < alpha

        self.friedman_test_results_ = friedman_test_results

    def calculate_adjusted_pvalues_results(self, control_oversampler=None):
        """Use the Holm's method to adjust the p-values of a paired difference
        t-test for every combination of classifiers and metrics using a control
        oversampler."""

        # Checks
        self._check_results()
        if not hasattr(self, 'wide_optimal_results_'):
            self.calculate_wide_optimal_results()

        # Get the oversamplers name
        oversamplers_names = self.wide_optimal_results_.columns[3:].tolist()
    
        # Use the last if no control oversampler is provided
        if control_oversampler is None:
            control_oversampler = oversamplers_names[-1]
        oversamplers_names.remove(control_oversampler)
    
        # Define empty p-values table
        pvalues = pd.DataFrame()

        # Populate p-values table
        for name in oversamplers_names:
            pvalues_pair = self.wide_optimal_results_.groupby(['Classifier', 'Metric'])[[name, control_oversampler]].apply(
                lambda df: ttest_rel(df[name], df[control_oversampler])[1])
            pvalues_pair = pd.DataFrame(pvalues_pair, columns=[name])
            pvalues = pd.concat([pvalues, pvalues_pair], axis=1)
    
        # Corrected p-values
        corrected_pvalues = pd.DataFrame(pvalues.apply(
            lambda col: multipletests(col, method='holm')[1], axis=1).values.tolist(), columns=oversamplers_names)
        corrected_pvalues = corrected_pvalues.set_index(pvalues.index).reset_index()
    
        self.adjusted_pvalues_results_ = corrected_pvalues
    
    def dump(self, path='.'):
        """Dump the object."""
        with open(join(path, f'{self.name}.pkl'), 'wb') as file:
            dump(self, file)

