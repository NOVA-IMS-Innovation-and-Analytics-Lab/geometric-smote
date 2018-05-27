"""
Test the data module.
"""

import pytest
from sklearn.datasets import make_regression
from ..validation import _check_param_grid, check_param_grids, check_datasets

X, y = make_regression()


def test_check_param_grid():
    param_grid = {'est__param': [1, 2, 3], 'est__param__subparam': [4, 5, 6]}
    updated_param_grid = param_grid.copy()
    updated_param_grid.update({'est_name': ['est']})
    assert _check_param_grid(param_grid) == updated_param_grid


def test_check_param_grids():
    param_grids = [{'est1__param': [1, 2, 3], 'est1__param__subparam': [4, 5, 6]},
                   {'est2__param': [10, 20, 30], 'est2__param__subparam': [40, 50, 60]}]
    updated_param_grids = [{'est1__param': [1, 2, 3], 'est1__param__subparam': [4, 5, 6], 'est_name': ['est1']},
                           {'est2__param': [10, 20, 30], 'est2__param__subparam': [40, 50, 60], 'est_name': ['est2']}]

    assert check_param_grids(param_grids) == updated_param_grids


@pytest.mark.parametrize('dataset_name1,dataset_name2,data', [
    ('ds1', 'ds1', (X, y)),
    ('ds1', X, (X, y)),
])
def test_check_datasets(dataset_name1, dataset_name2, data):
    with pytest.raises(ValueError):
        check_datasets([(dataset_name1, data), (dataset_name2, data)])