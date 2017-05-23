"""
This module contains classes to compare and evaluate 
the performance of various oversampling algorithms.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import pandas as pd
from os import listdir, chdir
from re import match, sub
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import make_scorer
from scipy.stats import friedmanchisquare


def count_elements(elements):
    """Returns a list with modified elements by a count index."""
    elements_map = {}
    elements_modified = []
    for element in elements:
        if element in elements_map.keys():
            elements_map[element] += 1
            elements_modified.append(element + str(elements_map[element]))
        else:
            elements_map[element] = 1
            elements_modified.append(element)
    return elements_modified

def extract_pvalue(dataframe):
    """Extracts p-value applying the Friedman test to measurements."""
    measurements = []
    for col in dataframe.columns[2:]:
        measurements.append(dataframe[col])
    return friedmanchisquare(*measurements).pvalue


class BinaryExperiment:
    """Class for comparison of oversampling algorithms performance 
    on imbalanced binary classification problems."""

    def __init__(self, 
        oversampling_methods, 
        classifiers,
        datasets,
        metrics=[roc_auc_score, f1_score, geometric_mean_score],
        n_splits=3, 
        experiment_repetitions=5, 
        random_state=None):
        self.oversampling_methods = oversampling_methods
        self.classifiers = classifiers
        self.metrics = metrics
        self.datasets = datasets
        self.n_splits = n_splits
        self.experiment_repetitions = experiment_repetitions
        self.random_state = random_state

    def _initialize_parameters(self):
        """Private method that initializes the experiment's parameters."""

        # Read csv files and save them to a dictionary
        if isinstance(self.datasets, str):
            chdir(self.datasets)
            self.datasets_ = {}
            csv_files = [csv_file for csv_file in listdir() if match('^.+\.csv$', csv_file)]
            for csv_file in csv_files:
                dataset = pd.read_csv(csv_file)
                X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
                dataset_name = sub(".csv", "", csv_file)
                self.datasets_[dataset_name] = (X, y)

        # If a list of (X,y) data is given, append names to each one of them
        if isinstance(self.datasets, list):
            self.datasets_ = {("dataset_" + str(ind + 1)):dataset for ind, dataset in enumerate(self.datasets)}
        
        # Create random states for experiments
        self.random_states_ = [self.random_state * index for index in range(self.experiment_repetitions)] if self.random_state is not None else [None] * self.experiment_repetitions
        
        # Extract names for experiments parameters
        self.classifiers_ = dict(zip(count_elements([classifier.__class__.__name__ for classifier in self.classifiers]), self.classifiers))
        self.oversampling_methods_ = dict(zip(count_elements([oversampling_method.__class__.__name__ if oversampling_method is not None else 'None' for oversampling_method in self.oversampling_methods]), self.oversampling_methods))
        self.metrics_ = dict(zip([sub('_', ' ', metric.__name__) for metric in self.metrics], self.metrics))
        
        # Converts metrics to scores
        self.scorers_ = dict(zip(self.metrics_.keys(), [make_scorer(metric) if metric is not roc_auc_score else make_scorer(metric, needs_threshold=True) for metric in self.metrics]))

    def _summarize_datasets(self):
        """Creates a summary of the datasets."""
        summary_columns = ["Dataset name", "# of features", "# of instances", "# of minority instances", "# of majority instances", "Imbalanced Ratio"]
        self.datasets_summary_ = pd.DataFrame({}, columns=summary_columns)
        for dataset_name, (X, y) in self.datasets_.items():
            n_instances = ((y == 0).sum(), (y == 1).sum())
            dataset_summary = pd.DataFrame([[dataset_name, X.shape[1], y.size, n_instances[1], n_instances[0], round(n_instances[0] / n_instances[1], 2)]], columns=self.datasets_summary_.columns)
            self.datasets_summary_ = self.datasets_summary_.append(dataset_summary, ignore_index=True)
        self.datasets_summary_[self.datasets_summary_.columns[1:-1]] = self.datasets_summary_[self.datasets_summary_.columns[1:-1]].astype(int)
        
    def run(self, logging_results=True):
        """Runs the experimental procedure and calculates the cross validation 
        scores for each classifier, oversampling method, datasets and metric."""
        self._initialize_parameters()
        self._summarize_datasets()
        
        # Populate results dataframe
        results = pd.DataFrame(columns=['Dataset', 'Classifier', 'Oversampling method', 'Metric', 'CV score'])
        for experiment_ind, random_state in enumerate(self.random_states_):
            cv = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
            for classifier_name, clf in self.classifiers_.items():
                clf.set_params(random_state=random_state)
                for dataset_name, (X, y) in self.datasets_.items():
                    for oversampling_method_name, oversampling_method in self.oversampling_methods_.items():
                        if oversampling_method is not None:
                            oversampling_method.set_params(random_state=random_state)
                            clf = make_pipeline(oversampling_method, clf)
                        for metric_name, scorer in self.scorers_.items():
                            cv_score = cross_val_score(clf, X, y, cv=cv, scoring=scorer).mean()
                            msg = 'Experiment: {}\nClassifier: {}\nOversampling method: {}\nMetric: {}\nDataset: {}\nCV score: {}\n\n'
                            if logging_results:
                                print(msg.format(experiment_ind + 1, classifier_name, oversampling_method_name, metric_name, dataset_name, cv_score))
                            result = pd.DataFrame([[dataset_name, classifier_name, oversampling_method_name, metric_name, cv_score]], columns=results.columns)
                            results = results.append(result, ignore_index=True)
        
        # Group results dataframe by dataset, classifier and metric
        grouped_results = results.groupby(list(results.columns[:-1]))

        # Calculate mean and std results
        self.mean_cv_results_ = grouped_results.mean().reset_index().rename(columns={'CV score': 'Mean CV score'})
        if self.experiment_repetitions > 1:
            self.std_cv_results_ = grouped_results.std().reset_index().rename(columns={'CV score': 'Std CV score'})
        else:
            self.std_cv_results_ = "Standard deviation is not calculated. More than one experiment repetition is needed."
        
        # Transform mean results to wide format
        mean_cv_results_wide = self.mean_cv_results_.pivot_table(index=['Dataset', 'Classifier', 'Metric'], columns=['Oversampling method'], values='Mean CV score').reset_index()
        mean_cv_results_wide.columns.rename(None, inplace=True)

        # Calculate mean ranking for each classifier/metric across datasets
        ranking_results = pd.concat([mean_cv_results_wide[['Classifier', "Metric"]], mean_cv_results_wide.apply(lambda row: len(row[3:]) - row[3:].argsort().argsort(), axis=1)], axis=1)    
        self.mean_ranking_results_ = round(ranking_results.groupby(['Classifier', 'Metric']).mean(), 2)

        # Calculate Friedman test p-values
        if len(self.oversampling_methods) > 2:
            self.friedman_test_results_ = ranking_results.groupby(['Classifier', 'Metric']).apply(extract_pvalue).reset_index().rename(columns={0: 'p-value'}).set_index(['Classifier', 'Metric'])
        else:
            self.friedman_test_results_ = 'Friedman test is not applied. More than two oversampling methods are needed.'