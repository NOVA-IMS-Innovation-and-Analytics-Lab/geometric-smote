"""
Test the classification module.
"""

import pytest
from ...metrics import weighted_mean_squared_error


@pytest.mark.parametrize('y_true,y_pred,asymmetry_factor', [
    ([1.0, 2.0, 3.0, 4.0], [2.0 , 3.0, 4.0, 5.0], 0.5),
    ([1.0, 2.0, 3.0, 4.0], [2.0 , 3.0, 2.0, 3.0], 0.0),
    ([1.0, 2.0, 3.0, 4.0], [0.0 , 1.0, 4.0, 5.0], 1.0)
])
def test_weighted_mean_squared_error(y_true, y_pred, asymmetry_factor):
    wmse = weighted_mean_squared_error(y_true, y_pred, asymmetry_factor=asymmetry_factor)
    assert wmse == 1.0





