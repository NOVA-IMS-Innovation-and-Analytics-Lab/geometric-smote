import pandas as pd
from os import listdir, chdir
from re import match, sub
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


class BinaryExperiment:
    """Class for comparison of oversampling algorithms performance 
    on imbalanced binary classification problems."""

    def __init__(self, 
        oversampling_methods, 
        classifiers, 
        metrics, 
        datasets,
        n_splits=5, 
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
        if isinstance(self.datasets, str):
            chdir(self.datasets)
            self.datasets = {}
            csv_files = [csv_file for csv_file in listdir() if match('^.+\.csv$', csv_file)]
            for csv_file in csv_files:
                dataset = pd.read_csv(csv_file)
                X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
                dataset_name = re.sub(".csv", "", csv_file)
                self.datasets[dataset_name] = (X, y)
        self.random_states_ = [self.random_state * index for index in range(self.experiment_repetitions)] if self.random_state is not None else [None] * self.experiment_repetitions
        self.cv_scores_ = []
        self.classifiers_names_ = [classifier.__class__.__name__ for classifier in self.classifiers]
        self.oversampling_methods_names_ = [oversampling_method.__class__.__name__ for oversampling_method in self.oversampling_methods]
        self.metrics_names_ = [metric.__name__ for metric in self.metrics]
        self.datasets_names_ = self.datasets.keys()
        
    def run(self):
        """Runs the experimental procedure and calculates the cross validation 
        scores for each classifier, oversampling method, datasets and metric."""
        self._initialize_parameters()
        for random_state in self.random_states_:
            cv = StratifiedKFold(n_splits=self.n_splits, random_state=random_state)
            for clf in self.classifiers:
                clf.set_params(random_state=random_state)
                for oversampling_method in self.oversampling_methods:
                    oversampling_method.set_params(random_state=random_state)
                    for metric in self.metrics:
                        for X, y in self.datasets:
                            if oversampling_method is not None:
                                clf = make_pipeline(oversampling_method, clf)
                            self.cv_scores_.append(cross_val_score(clf, X, y, cv=cv, scoring=metric).mean())

    def get_mean_results():
        pass

    def get_std_results():
        pass

    def get_ranking_results():
        pass

    def get_friedman_test_results():
        pass
