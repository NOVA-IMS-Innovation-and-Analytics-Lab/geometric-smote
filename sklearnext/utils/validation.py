"""
The :mod:`sklearnext.utils.validation` includes utilities
for input validation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from sklearn.utils import check_random_state


def _check_param_grids(param_grids):
    """Normalize the parameter grid to use with
    parametrized estimators."""
    normalized_param_grids = param_grids.copy()
    est_name = list(set([param.split('__')[0] for param in param_grids.keys()]))
    normalized_param_grids.update({'est_name': est_name})
    return normalized_param_grids


def check_param_grids(param_grids):
    """Normalize the parameter grid to use with
    parametrized estimators."""
    if isinstance(param_grids, list):
        normalized_param_grids = []
        for param_grid in param_grids:
            normalized_param_grids.append(_check_param_grids(param_grid))
    else:
        normalized_param_grids = _check_param_grids(param_grids)
    return normalized_param_grids


def check_datasets(datasets):
    """Check that datasets is a list of (X,y) pairs or a dictionary of dataset-name:(X,y) pairs."""
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
    """Create random states for experiments."""
    random_state = check_random_state(random_state)
    return [random_state.randint(0, 2 ** 32 - 1, dtype='uint32') for _ in range(repetitions)]




