"""
This module contains classes to compare and evaluate 
the performance of various oversampling algorithms.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import make_scorer
from sklearn.base import clone
from imblearn.pipeline import Pipeline
from imblearn.metrics import geometric_mean_score
from .utils import check_datasets, check_random_states
from os.path import join
from os import listdir
from re import match, sub
from scipy.stats import friedmanchisquare
from progressbar import ProgressBar


def read_csv_dir(filepath):
    "Reads a directory of csv files and returns a dictionary of dataset-name:(X,y) pairs."
    datasets = {}
    csv_files = [csv_file for csv_file in listdir(filepath) if match('^.+\.csv$', csv_file)]
    for csv_file in csv_files:
        dataset = pd.read_csv(join(filepath,csv_file))
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        dataset_name = sub(".csv", "", csv_file)
        datasets[dataset_name] = (X, y)
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

def optimize_hyperparameters(X, y, clf, param_grid, cv):
    """Returns the parameters with the highest auc."""
    clfs = GridSearchCV(estimator=clone(clf), param_grid=param_grid, scoring='roc_auc', cv=cv, refit=False)
    clfs.fit(X, y)
    return clfs.best_params_


class BinaryExperiment:
    """Class for comparison of oversampling algorithms performance 
    on imbalanced binary classification problems.
    
    Parameters
    ----------
    datasets : str or list of (X, y) tuples or dictionary of dataset-name:(X,y) pairs
        The string is a path to the directory which contains the imbalanced data in 
        csv format. The list of (X, y) pairs is a list of tuples of input data and 
        target values, The dictionary extends the list by adding the datasets names 
        as a key.
    classifiers : list of classifiers
        A list of classifiers.
    oversampling_methods : list of oversampling_methods
        A list of oversampling methods.
    metrics : list of metrics, (default=[roc_auc_score, f1_score, geometric_mean_score])
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
    param_grids : list, optional (default=None)
        A list of hyperparameters grids for each classfier.
    n_jobs : int, (default=1)
        The number of CPUs to use to do the computation. -1 means ‘all CPUs’.
    """

    def __init__(self, 
                 datasets,
                 classifiers,
                 oversampling_methods, 
                 metrics=[roc_auc_score, f1_score, geometric_mean_score],
                 n_splits=3, 
                 experiment_repetitions=5, 
                 random_state=None, 
                 param_grids=None,
                 n_jobs=1):
        self.datasets = datasets
        self.classifiers = classifiers
        self.oversampling_methods = oversampling_methods
        self.metrics = metrics
        self.n_splits = n_splits
        self.experiment_repetitions = experiment_repetitions
        self.random_state = random_state
        self.param_grids = param_grids
        self.n_jobs = n_jobs

    def _initialize_parameters(self):
        """Private method that initializes the experiment's parameters."""
        
        self.datasets_ = check_datasets(self.datasets)
        self.random_states_ = check_random_states(self.random_state, self.experiment_repetitions)
        

        # Extract names for experiments parameters
        self.classifiers_ = self.classifiers
        self.oversampling_methods_ = self.oversampling_methods
        self.metrics_ = dict(zip([sub('_', ' ', metric.__name__) for metric in self.metrics], self.metrics))
        
            
        # Converts metrics to scores
        self.scorers_ = dict(zip(self.metrics_.keys(), [make_scorer(metric) if metric is not roc_auc_score else make_scorer(metric, needs_threshold=True) for metric in self.metrics]))
        
    def run(self):
        """Runs the experimental procedure and calculates the cross validation 
        scores for each classifier, oversampling method, datasets and metric."""
        self._initialize_parameters()
        bar = ProgressBar(redirect_stdout=True, max_value=len(self.random_states_) * len(self.datasets_) * len(self.classifiers_) * len(self.oversampling_methods_) * len(self.metrics_))
        iterations = 0

        # Populate results dataframe
        results_columns = ['Dataset', 'Classifier', 'Oversampling method', 'Metric', 'CV score']
        results = pd.DataFrame(columns=results_columns)
        for experiment_ind, random_state in enumerate(self.random_states_):
            cv = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
            for dataset_name, (X, y) in self.datasets_.items():
                for classifier_name, clf, _ in self.classifiers_:
                    if 'random_state' in clf.get_params().keys():
                        clf.set_params(random_state=random_state)
                    for oversampling_method_name, oversampling_method, _ in self.oversampling_methods_:
                        if oversampling_method is not None:
                            oversampling_method.set_params(random_state=random_state)
                            clf = Pipeline([(oversampling_method_name, oversampling_method), (classifier_name, clf)])
                        cv_output = cross_validate(clf, X, y, cv=cv, scoring=self.scorers_, n_jobs=self.n_jobs)
                        for metric_name, scorer in self.scorers_.items():
                            cv_score = cv_output["test_" + metric_name].mean()
                            result_list = [dataset_name, classifier_name, oversampling_method_name, metric_name, cv_score]
                            result = pd.DataFrame([result_list], columns=results_columns)
                            results = results.append(result, ignore_index=True)
                            iterations += 1
                            bar.update(iterations)

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