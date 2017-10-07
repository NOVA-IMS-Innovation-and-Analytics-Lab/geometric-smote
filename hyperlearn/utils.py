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
    paramspace = _ParamSpace()
    nums_subspaces = []
    for est_name, _, *param_grid in estimator:
        ps = _ParamSpace(param_grid[0] if len(param_grid) > 0 else None)
        nums_subspaces.append(ps.num_subspaces)
        ps.append_prefix(est_name)
        paramspace *= ps
    if len(set(nums_subspaces)) > 1:
        raise ValueError('The lists of parameter grids for all pipeline\'s estimators should have the same length or missing.')
    return paramspace.param_grid if paramspace.param_grid != [] else {}

def check_param_grids(estimators):
    """Parses the parameter grids."""
    return [_check_param_grid(estimator) for estimator in estimators]

class _ParamSpace:
    """Private class for the creation and modification of 
    a hyperspace.
    """

    def __init__(self, param_grid=None):
        self.param_grid = param_grid if param_grid is not None else []
    
    def __add__(self, paramspace):
        ps = _ParamSpace()
        ps.param_grid = self.param_grid + paramspace.param_grid
        return ps

    def __mul__(self, paramspace):
        ps = _ParamSpace()
        if self.param_grid != [] and paramspace.param_grid != []:
            for zipped_grid in zip(self.param_grid, paramspace.param_grid):
                param_grid = {}
                for grid in zipped_grid:
                    param_grid.update(grid)
                ps.param_grid.append(param_grid)
        elif self.param_grid == []:
            ps.param_grid = paramspace.param_grid
        elif paramspace.param_grid == []:
            ps.param_grid = self.param_grid
        return ps

    def append_prefix(self, est_name):
        self.param_grid = [{(est_name + '__' + k):v for k, v in param_grid.items()} for param_grid in self.param_grid]
        
    @property
    def num_subspaces(self):
        return len(self.param_grid)