"""
This module contains various checks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from sklearn.utils import check_X_y, check_random_state
from imblearn.pipeline import Pipeline


def check_datasets(datasets):
    """Checks that datasets is a list of (X,y) pairs or a dictionary of dataset-name:(X,y) pairs."""
    try:
        datasets_names = [dataset_name for dataset_name, _ in datasets]
        are_all_strings = all([isinstance(dataset_name, str) for dataset_name in datasets_names])
        are_unique = len(list(datasets_names)) == len(set(datasets_names))
        if are_all_strings and are_unique:
            return [(dataset_name, check_X_y(*dataset)) for dataset_name, dataset in datasets]
        else:
            raise ValueError("The datasets' names should be unique strings.")
    except:
        raise ValueError("The datasets should be a list (dataset name:(X,y)) pairs.")

def check_random_states(random_state, repetitions):
    """Creates random states for experiments."""
    random_state = check_random_state(random_state)
    return [random_state.randint(0, 2 ** 32 - 1, dtype='uint32') for ind in range(repetitions)]

def check_estimators(estimators):
    """Parses the pipelines of transformations and estimators."""
    return [Pipeline([(name, est) for name, est, *_ in estimator]) for estimator in estimators]

def _check_param_grid(estimator):
    grids = []
    for est_name, _, *param_grid in estimator:
        if param_grid != []:
            grids.append([{(est_name + '__' + k):v for k, v in grid.items()} for grid in param_grid[0]])
    grids_length = [len(param_grid) for param_grid in grids]
    if len(set(grids_length)) > 1:
        raise ValueError('The lists of parameter grids for all pipeline\'s estimators should have the same length or not be defined.')
    param_grid = []
    for params in zip(*grids):
        parameters = {}
        for par in params:
            parameters.update(par)
        param_grid.append(parameters)
    return param_grid

def check_param_grids(estimators):
    """Parses the parameter grids."""
    return [_check_param_grid(estimator) for estimator in estimators]
