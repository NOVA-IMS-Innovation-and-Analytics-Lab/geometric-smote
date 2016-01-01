"""
This module contains classes to compare and evaluate 
the performance of various oversampling algorithms.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_validate
from imblearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score
from .utils import check_datasets, check_random_states, check_models
from os.path import join
from os import listdir
from re import match, sub
from scipy.stats import friedmanchisquare
from progressbar import ProgressBar


def read_csv_dir(filepath):
    "Reads a directory of csv files and returns a dictionary of dataset-name:(X,y) pairs."
    datasets = []
    csv_files = [csv_file for csv_file in listdir(filepath) if match('^.+\.csv$', csv_file)]
    for csv_file in csv_files:
        dataset = pd.read_csv(join(filepath,csv_file))
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        dataset_name = sub(".csv", "", csv_file)
        datasets.append( (dataset_name, (X, y)) )
    return datasets

def summarize_datasets(datasets):
        """Creates a summary of the datasets."""
        datasets = check_datasets(datasets)
        summary_columns = ["Dataset name", "# of features", "# of instances", "# of minority instances", "# of majority instances", "Imbalance Ratio"]
        datasets_summary = pd.DataFrame({}, columns=summary_columns)
        for dataset_name, (X, y) in datasets.items():
            n_instances = ((y == 0).sum(), (y == 1).sum())
            dataset_summary = pd.DataFrame([[dataset_name, X.shape[1], y.size, n_instances[1], n_instances[0], round(n_instances[0] / n_instances[1], 2)]], columns=datasets_summary.columns)
            datasets_summary = datasets_summary.append(dataset_summary, ignore_index=True)
        datasets_summary[datasets_summary.columns[1:-1]] = datasets_summary[datasets_summary.columns[1:-1]].astype(int)
        return datasets_summary

def extract_pvalue(dataframe):
    """Extracts p-value applying the Friedman test to measurements."""
    measurements = []
    for col in dataframe.columns[2:]:
        measurements.append(dataframe[col])
    return friedmanchisquare(*measurements).pvalue


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
    
    def run(self):
        """Runs the experimental procedure and calculates the cross validation 
        scores for each classifier, oversampling method, datasets and metric."""
        datasets = check_datasets(self.datasets)
        self.datasets_names_ = [dataset_name for dataset_name, _ in datasets]
        self.random_states_ = check_random_states(self.random_state, self.experiment_repetitions)
        self.classifiers_ = check_models(self.classifiers, "classifier")
        self.oversamplers_ = check_models(self.oversamplers, "oversampler")
        bar = ProgressBar(redirect_stdout=True, max_value=len(self.random_states_) * len(datasets) * len(self.classifiers_) * len(self.oversamplers_))
        iterations = 0

        # Populate results dataframe
        results_columns = ['Dataset', 'Classifier', 'Oversampler', 'Metric', 'CV score']
        self.results_ = pd.DataFrame(columns=results_columns)
        for experiment_ind, random_state in enumerate(self.random_states_):
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
                        bar.update(iterations)
                        for scorer in self.scoring:
                            cv_score = cv_output["test_" + scorer].mean()
                            result_list = [dataset_name, clf_name, oversampler_name, scorer, cv_score]
                            result = pd.DataFrame([result_list], columns=results_columns)
                            self.results_ = self.results_.append(result, ignore_index=True)
                            
        # Group results dataframe by dataset, classifier and metric
        grouped_results = self.results_.groupby(list(self.results_.columns[:-1]))

        # Calculate mean and std results
        self.mean_cv_results_ = grouped_results.mean().reset_index().rename(columns={'CV score': 'Mean CV score'})
        if self.experiment_repetitions > 1:
            self.std_cv_results_ = grouped_results.std().reset_index().rename(columns={'CV score': 'Std CV score'})
        else:
            self.std_cv_results_ = "Standard deviation is not calculated. More than one experiment repetition is needed."
        
        # Transform mean results to wide format
        mean_cv_results_wide = self.mean_cv_results_.pivot_table(index=['Dataset', 'Classifier', 'Metric'], columns=['Oversampler'], values='Mean CV score').reset_index()
        mean_cv_results_wide.columns.rename(None, inplace=True)

        # Calculate mean ranking for each classifier/metric across datasets
        ranking_results = pd.concat([mean_cv_results_wide[['Classifier', "Metric"]], mean_cv_results_wide.apply(lambda row: len(row[3:]) - row[3:].argsort().argsort(), axis=1)], axis=1)    
        self.mean_ranking_results_ = round(ranking_results.groupby(['Classifier', 'Metric']).mean(), 2)

        # Calculate Friedman test p-values
        if len(self.oversamplers) > 2:
            self.friedman_test_results_ = ranking_results.groupby(['Classifier', 'Metric']).apply(extract_pvalue).reset_index().rename(columns={0: 'p-value'}).set_index(['Classifier', 'Metric'])
        else:
            self.friedman_test_results_ = 'Friedman test is not applied. More than two oversampling methods are needed.'
        
        del self.datasets