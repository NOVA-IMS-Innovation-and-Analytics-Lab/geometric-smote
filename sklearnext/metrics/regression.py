"""
The :mod:`sklearnext.metrics.regressio` contains
various metrics for regression tasks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import numpy as np
from sklearn.metrics.regression import _check_reg_targets, check_consistent_length
from sklearn.externals.six import string_types


def weighted_mean_squared_error(y_true,
                                y_pred,
                                sample_weight=None,
                                multioutput='uniform_average',
                                asymmetry_factor=0.5):
    """Weighted mean squared error regression loss

    Parameters
    ----------
    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape = (n_samples), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average']
        or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
    asymmetry_factor : float
        Asymmetry factor between underprediction and overprediction
        in the range [0.0, 0.1].  The balanced case corresponds to the 0.5 value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """

    # Check input data
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    
    # Calculate weights
    weights = abs((y_true < y_pred) - asymmetry_factor)

    # Calculate errors
    output_errors = 2* np.average(weights * (y_true - y_pred) ** 2, axis=0, weights=sample_weight)
    
    # Handle multioutput
    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            multioutput = None
    
    return np.average(output_errors, weights=multioutput)
