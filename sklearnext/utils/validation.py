"""
The :mod:`sklearnext.utils.validation` includes utilities
for input validation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from itertools import product
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.model_selection._search import _check_param_grid
from imblearn.pipeline import Pipeline


def _normalize_param_grid(param_grid):
    """Normalize the parameter grid to use with
    parametrized estimators."""
    _check_param_grid(param_grid)
    normalized_param_grid = param_grid.copy()
    est_name = list(set([param.split('__')[0] for param in param_grid.keys()]))
    normalized_param_grid.update({'est_name': est_name})
    return normalized_param_grid


def check_param_grids(param_grids, estimators):
    """Normalize the parameter grid to use with
    parametrized estimators."""
    if isinstance(param_grids, list):
        normalized_param_grids = []
        for param_grid in param_grids:
            normalized_param_grids.append(_normalize_param_grid(param_grid)
                                          if 'est_name' not in param_grid.keys() else param_grid.copy())
    else:
        normalized_param_grids = [_normalize_param_grid(param_grids)
                                  if 'est_name' not in param_grids.keys() else param_grids.copy()]
    est_names, _ = zip(*estimators)
    est_names = set(est_names)
    try:
        generated_est_names = set([param_grid['est_name'][0] for param_grid in normalized_param_grids])
    except IndexError:
        generated_est_names = set()
        normalized_param_grids = []
    for est_name in est_names.difference(generated_est_names):
        normalized_param_grids += [{'est_name': [est_name]}]
    return normalized_param_grids


def check_oversamplers_classifiers(oversamplers, classifiers, n_runs, random_state, n_datasets):
    """Extract estimators and parameters grids."""

    # Extract estimators
    estimators_products = product(
        [smpl[0:2] for smpl in oversamplers],
        [clf[0:2] for clf in classifiers],
        range(n_runs),
        range(n_datasets)
    )
    estimators = [('%s_%s_%s_%s' % (smpl_name, clf_name, run_id, dataset_id), Pipeline([(smpl_name, smpl), (clf_name, clf)])) for
                  (smpl_name, smpl), (clf_name, clf), run_id, dataset_id in estimators_products]

    # Extract parameters grids
    oversamplers_param_grids = [{('%s__%s' % (smpl[0], par)):val for par, val in smpl[2].items()}
                                if len(smpl) > 2 else {} for smpl in oversamplers]
    classifiers_param_grids = [{('%s__%s' % (clf[0], par)): val for par, val in clf[2].items()}
                               if len(clf) > 2 else {} for clf in classifiers]
    param_grids_products = product(oversamplers_param_grids, classifiers_param_grids, range(n_runs), range(n_datasets))
    param_grids = []
    est_names, _ = zip(*estimators)
    for (oversampler_param_grid , classifier_param_grid, _, dataset_id), est_name in zip(param_grids_products, est_names):
        param_grid = {}
        param_grid.update(oversampler_param_grid)
        param_grid.update(classifier_param_grid)
        param_grid = {('%s__%s' % (est_name, par)):val for par, val in param_grid.items()}
        param_grid.update({'est_name': [est_name]})
        param_grid.update({'dataset_id': [dataset_id if n_datasets > 1 else None]})
        param_grids.append(param_grid)

    # Extract random states
    random_states = check_random_states(random_state, len(oversamplers) * len(classifiers) * n_runs * n_datasets)
    zipped_param_grids = zip(param_grids, random_states)
    smpl_clf_param_grids = []
    for param_grid, random_state in zipped_param_grids:
        smpl_clf_param_grid = param_grid.copy()
        smpl_clf_param_grid.update({'random_state': [random_state]})
        smpl_clf_param_grids.append(smpl_clf_param_grid)

    return {'estimators': estimators, 'param_grids': smpl_clf_param_grids}


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


def check_estimators(estimators):
    """Check estimators correct input."""
    error_msg = 'Invalid `estimators` attribute, `estimators` should be a list of (string, estimator) tuples.'
    try:
        if not all([all([isinstance(name, str), isinstance(est, BaseEstimator)]) for name, est in estimators]) or len(estimators) == 0:
            raise AttributeError(error_msg)
    except:
        raise AttributeError(error_msg)



