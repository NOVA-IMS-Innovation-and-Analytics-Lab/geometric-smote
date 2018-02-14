"""
This module contains various checks.
"""

from sklearn.model_selection import ParameterGrid
from imblearn.pipeline import Pipeline


def check_pipelines(pipelines):
    """Parses the pipelines of transformations and estimators."""
    return [Pipeline([(name, est) for name, est, *_ in pipeline]) for pipeline in pipelines]

def _check_param_grid(pipelines):
    prefixes = [prefix for prefix, _, *_ in pipelines]
    param_grids = [param_grid[0] if param_grid else None for _, _, *param_grid in pipelines]
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
        raise ValueError('Non compatible parameter grids for all estimators.')
    flat_param_grid = []
    for i in range(param_grids_length[0]):
        param_grid = [param_grid[i] for param_grid in param_grids]
        param_grid_dict = {}
        for prefix, sub_param_grid in zip(prefixes, param_grid):
            param_grid_dict.update({prefix + '__' + p:v for p, v in sub_param_grid.items()})
        flat_param_grid.append(param_grid_dict)
    flat_param_grid = [{p:[v] for p, v in params.items()}\
                       for params in list(ParameterGrid(flat_param_grid))]
    return flat_param_grid

def check_param_grids(pipelines):
    """Parses the parameter grids."""
    return [_check_param_grid(pipeline) for pipeline in pipelines]
