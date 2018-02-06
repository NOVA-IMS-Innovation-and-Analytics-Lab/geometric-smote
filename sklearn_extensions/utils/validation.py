"""
Utilities for input validation
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from sklearn.utils import check_X_y, check_random_state
from sklearn.model_selection import ParameterGrid
from imblearn.pipeline import Pipeline


def check_random_states():
    pass


def check_datasets(datasets):
    """Checks that datasets is a list of (X,y) pairs or a dictionary of dataset-name:(X,y) pairs."""
    try:
        datasets_names = [dataset_name for dataset_name, _ in datasets]
        are_all_strings = all([isinstance(dataset_name, str) for dataset_name in datasets_names])
        are_unique = len(list(datasets_names)) == len(set(datasets_names))
        if are_all_strings and are_unique:
            return datasets
        else:
            raise ValueError("The datasets' names should be unique strings.")
    except:
        raise ValueError("The datasets should be a list of (dataset name:(X,y)) pairs.")


def check_random_states(random_state, repetitions):
    """Creates random states for experiments."""
    random_state = check_random_state(random_state)
    return [random_state.randint(0, 2 ** 32 - 1, dtype='uint32') for ind in range(repetitions)]


def check_estimators(estimators):
    """Parses the pipelines of transformations and estimators."""
    return [Pipeline([(name, est) for name, est, *_ in estimator]) for estimator in estimators]


def _check_param_grid(estimator_tpls):
    prefixes = [prefix for prefix, _, *_ in estimator_tpls]
    param_grids = [param_grid[0] if param_grid else None for _, _, *param_grid in estimator_tpls]
    remove_prefixes, remove_param_grids = [], []
    for prefix, param_grid in zip(prefixes, param_grids):
        if param_grid is None:
            remove_prefixes.append(prefix)
            remove_param_grids.append(param_grid)
    prefixes = [prefix for prefix in prefixes if prefix not in remove_prefixes]
    param_grids = [param_grid for param_grid in param_grids if param_grid not in remove_param_grids]
    if not param_grids:
        return {}
    param_grids_length = [len(param_grid) for param_grid in param_grids]
    if len(set(param_grids_length)) > 1:
        raise ValueError('Parameter grids for all the estimators should have either the same length or not defined.')
    flat_param_grid = []
    for i in range(param_grids_length[0]):
        param_grid = [param_grid[i] for param_grid in param_grids]
        param_grid_dict = {}
        for prefix, sub_param_grid in zip(prefixes, param_grid):
            param_grid_dict.update({prefix + '__' + p: v for p, v in sub_param_grid.items()})
        flat_param_grid.append(param_grid_dict)
    flat_param_grid = [{p: [v] for p, v in params.items()} for params in list(ParameterGrid(flat_param_grid))]
    return flat_param_grid


def check_param_grids(estimators):
    """Parses the parameter grids."""
    return [_check_param_grid(estimator) for estimator in estimators]
