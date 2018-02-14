"""
This module contains various metrics
for classification and regression tasks.
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.regression import _check_reg_targets, check_consistent_length
from sklearn.externals.six import string_types


def weighted_mean_squared_error(y_true,
                                y_pred,
                                sample_weight=None,
                                multioutput='uniform_average',
                                asymmetry_factor=0.1):
    """Weighted mean squared error regression loss.

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
        Assymetry factor between underprediction and overprediction
        in the range [0.0, 0.1].  The balanced case corresponds to the 0.5 value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    weights = abs((y_true < y_pred) - asymmetry_factor)
    output_errors = 2 * np.average(weights * (y_true - y_pred) ** 2, axis=0, weights=sample_weight)
    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            multioutput = None
    return np.average(output_errors, weights=multioutput)

def weighted_mean_absolute_error(y_true,
                                 y_pred,
                                 sample_weight=None,
                                 multioutput='uniform_average',
                                 asymmetry_factor=0.2):
    """Weighted mean absolute error regression loss.

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
        Assymetry factor between underprediction and overprediction
        in the range [0.0, 0.1].  The balanced case corresponds to the 0.5 value.

    Returns
    -------
    loss : float or ndarray of floats
        A non-negative floating point value (the best value is 0.0), or an
        array of floating point values, one for each individual target.
    """
    _, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    check_consistent_length(y_true, y_pred, sample_weight)
    weights = abs((y_true < y_pred) - asymmetry_factor)
    output_errors = 2 * np.average(weights * np.abs(y_true - y_pred), axis=0, weights=sample_weight)
    if isinstance(multioutput, string_types):
        if multioutput == 'raw_values':
            return output_errors
        elif multioutput == 'uniform_average':
            multioutput = None
    return np.average(output_errors, weights=multioutput)

def tp_score(y_true, y_pred, labels=None, sample_weight=None):
    """True positive score function.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : int
        Returns the number of correctly classified positive
        class samples.
    """
    return confusion_matrix(y_true, y_pred, labels, sample_weight)[1, 1]

def tn_score(y_true, y_pred, labels=None, sample_weight=None):
    """True negative score function.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : int
        Returns the number of correctly classified negative
        class samples.
    """
    return confusion_matrix(y_true, y_pred, labels, sample_weight)[0, 0]

def fp_score(y_true, y_pred, labels=None, sample_weight=None):
    """False positive score function.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : int
        Returns the number of incorrectly classified negative
        class samples as positive class samples.
    """
    return confusion_matrix(y_true, y_pred, labels, sample_weight)[0, 1]

def fn_score(y_true, y_pred, labels=None, sample_weight=None):
    """False negative score function.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (correct) target values.
    y_pred : array, shape = [n_samples]
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_classes], optional
        List of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
        If none is given, those that appear at least once
        in ``y_true`` or ``y_pred`` are used in sorted order.
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    Returns
    -------
    score : int
        Returns the number of incorrectly classified positive
        class samples as negative class samples.
    """
    return confusion_matrix(y_true, y_pred, labels, sample_weight)[1, 0]
