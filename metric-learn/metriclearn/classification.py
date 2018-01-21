"""
This module contains various metrics for imbalanced classification tasks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from sklearn.metrics import confusion_matrix


def tp_score(y_true, y_pred):
    """True positive score function."""
    return confusion_matrix(y_true, y_pred)[1, 1]

def tn_score(y_true, y_pred):
    """True negative score function."""
    return confusion_matrix(y_true, y_pred)[0, 0]

def fp_score(y_true, y_pred):
    """False positive score function."""
    return confusion_matrix(y_true, y_pred)[0, 1]

def fn_score(y_true, y_pred):
    """False negative score function."""
    return confusion_matrix(y_true, y_pred)[1, 0]
