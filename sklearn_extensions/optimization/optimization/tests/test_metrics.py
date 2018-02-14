"""
Test the metrics module.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error
import pytest
from ..metrics import (
    weighted_mean_squared_error,
    weighted_mean_absolute_error,
    tp_score,
    tn_score,
    fp_score,
    fn_score)


Y_TRUE_BIN = [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
Y_PRED_BIN = [1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
Y_TRUE_REG = [3.0, 4.0, 5.0]
Y_PRED_REG_UNDER = [2.0, 3.0, 4.0]
Y_PRED_REG_OVER = [4.0, 5.0, 6.0]

@pytest.mark.parametrize('y_true,y_pred,asymmetry_factor', [
    (Y_TRUE_REG, Y_PRED_REG_UNDER, 0.5),
    (Y_TRUE_REG, Y_PRED_REG_UNDER, 0.2),
    (Y_TRUE_REG, Y_PRED_REG_UNDER, 0.8),
    (Y_TRUE_REG, Y_PRED_REG_OVER, 0.5),
    (Y_TRUE_REG, Y_PRED_REG_OVER, 0.2),
    (Y_TRUE_REG, Y_PRED_REG_OVER, 0.8)
])
def test_weighted_mean_squared_error(y_true, y_pred, asymmetry_factor):
    """Test weighted mean squared error."""
    wmse = weighted_mean_squared_error(y_true, y_pred, asymmetry_factor=asymmetry_factor)
    mse = mean_squared_error(y_true, y_pred)
    if asymmetry_factor == 0.5:
        assert wmse == mse
    elif asymmetry_factor < 0.5:
        if y_pred == 'Y_PRED_REG_UNDER':
            assert wmse < mse
        if y_pred == 'Y_PRED_REG_OVER':
            assert wmse > mse
    elif asymmetry_factor > 0.5:
        if y_pred == 'Y_PRED_REG_UNDER':
            assert wmse > mse
        if y_pred == 'Y_PRED_REG_OVER':
            assert wmse < mse

@pytest.mark.parametrize('y_true,y_pred,asymmetry_factor', [
    (Y_TRUE_REG, Y_PRED_REG_UNDER, 0.5),
    (Y_TRUE_REG, Y_PRED_REG_UNDER, 0.2),
    (Y_TRUE_REG, Y_PRED_REG_UNDER, 0.8),
    (Y_TRUE_REG, Y_PRED_REG_OVER, 0.5),
    (Y_TRUE_REG, Y_PRED_REG_OVER, 0.2),
    (Y_TRUE_REG, Y_PRED_REG_OVER, 0.8)
])
def test_weighted_mean_absolute_error(y_true, y_pred, asymmetry_factor):
    """Test weighted mean absolute error."""
    wmae = weighted_mean_absolute_error(y_true, y_pred, asymmetry_factor=asymmetry_factor)
    mae = mean_absolute_error(y_true, y_pred)
    if asymmetry_factor == 0.5:
        assert wmae == mae
    elif asymmetry_factor < 0.5:
        if y_pred == 'Y_PRED_REG_UNDER':
            assert wmae < mae
        if y_pred == 'Y_PRED_REG_OVER':
            assert wmae > mae
    elif asymmetry_factor > 0.5:
        if y_pred == 'Y_PRED_REG_UNDER':
            assert wmae > mae
        if y_pred == 'Y_PRED_REG_OVER':
            assert wmae < mae

def test_tp_score():
    """Test true positive score."""
    assert tp_score(Y_TRUE_BIN, Y_PRED_BIN) == 3

def test_tn_score():
    """Test true negative score."""
    assert tn_score(Y_TRUE_BIN, Y_PRED_BIN) == 2

def test_fp_score():
    """Test false positive score."""
    assert fp_score(Y_TRUE_BIN, Y_PRED_BIN) == 2

def test_fn_score():
    """Test false negative score."""
    assert fn_score(Y_TRUE_BIN, Y_PRED_BIN) == 3
