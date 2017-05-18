"""
This module contains various evaluation metrics 
for binary classification problems.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from math import sqrt


def confusion_matrix_values(y_true, y_pred, pos_label):
    """Helper function that returns the confusion matrix elements 
    for ground truth target labels y_true and predicted target 
    labels by a classifier y_pred. The positive class is represented 
    by pos_label.
    
    Parameters
    ----------
    y_true : 1d array numpy array
        Ground truth target values.
    
    y_pred : 1d array numpy array
        Predicted target labels as returned by a classifier.

    pos_label : int, 1 by default
        The positive class. 
    """

    true_positive = (y_pred == pos_label)[y_true == pos_label].sum()
    false_positive = (y_pred == pos_label)[y_true != pos_label].sum()
    true_negative = (y_pred != pos_label)[y_true != pos_label].sum()
    false_negative = (y_pred != pos_label)[y_true == pos_label].sum()
    return true_positive, false_positive, true_negative, false_negative

def precision(y_true, y_pred, pos_label=1):
    """Returns the precision score for ground truth target labels 
    y_true and predicted target labels by a classifier y_pred. The 
    positive class is represented by pos_label.
    
    Parameters
    ----------
    y_true : 1d array numpy array
        Ground truth target values.
    
    y_pred : 1d array numpy array
        Predicted target labels as returned by a classifier.

    pos_label : int, 1 by default
        The positive class.
    """

    true_positive, false_positive, _, _ = confusion_matrix_values(y_true, y_pred, pos_label)
    return true_positive / (true_positive + false_positive)

def recall(y_true, y_pred, pos_label=1):
    """Returns the recall score for ground truth target labels 
    y_true and predicted target labels by a classifier y_pred. The 
    positive class is represented by pos_label.
    
    Parameters
    ----------
    y_true : 1d array numpy array
        Ground truth target values.
    
    y_pred : 1d array numpy array
        Predicted target labels as returned by a classifier.

    pos_label : int, 1 by default
        The positive class.
    """

    true_positive, _, _, false_negative = confusion_matrix_values(y_true, y_pred, pos_label)
    return true_positive / (true_positive + false_negative)

def sensitivity(y_true, y_pred, pos_label=1):
    """Returns the sensitivity for ground truth target labels 
    y_true and predicted target labels by a classifier y_pred. The 
    positive class is represented by pos_label.
    
    Parameters
    ----------
    y_true : 1d array numpy array
        Ground truth target values.
    
    y_pred : 1d array numpy array
        Predicted target labels as returned by a classifier.

    pos_label : int, 1 by default
        The positive class.
    """

    true_positive, _, _, false_negative = confusion_matrix_values(y_true, y_pred, pos_label)
    return true_positive / (true_positive + false_negative)

def specificity(y_true, y_pred, pos_label=1):
    """Returns the specificity for ground truth target labels 
    y_true and predicted target labels by a classifier y_pred. The 
    positive class is represented by pos_label.
    
    Parameters
    ----------
    y_true : 1d array numpy array
        Ground truth target values.
    
    y_pred : 1d array numpy array
        Predicted target labels as returned by a classifier.

    pos_label : int, 1 by default
        The positive class.
    """

    _, false_positive, true_negative, _ = confusion_matrix_values(y_true, y_pred, pos_label)
    return true_negative / (true_negative + false_positive)

def F_measure(y_true, y_pred, pos_label=1):
    """Returns the balanced F-measure for ground truth target labels 
    y_true and predicted target labels by a classifier y_pred. The 
    positive class is represented by pos_label.
    
    Parameters
    ----------
    y_true : 1d array numpy array
        Ground truth target values.
    
    y_pred : 1d array numpy array
        Predicted target labels as returned by a classifier.

    pos_label : int, 1 by default
        The positive class.
    """

    true_positive, false_positive, true_negative, false_negative = confusion_matrix_values(y_true, y_pred, pos_label)
    return 2 * true_positive / (2 * true_positive + false_positive + false_negative)

def G_mean(y_true, y_pred, pos_label=1):
    """Returns the G-mean for ground truth target labels y_true and 
    predicted target labels by a classifier y_pred. The positive 
    class is represented by pos_label.
    
    Parameters
    ----------
    y_true : 1d array numpy array
        Ground truth target values.
    
    y_pred : 1d array numpy array
        Predicted target labels as returned by a classifier.

    pos_label : int, 1 by default
        The positive class.
    """

    return sqrt(sensitivity(y_true, y_pred, pos_label) * specificity(y_true, y_pred, pos_label)) 