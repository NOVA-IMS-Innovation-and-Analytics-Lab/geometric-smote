"""
The :mod:`sklearnext.metrics.classification` contains
various metrics for classification tasks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from sklearn.metrics import make_scorer, f1_score, confusion_matrix


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


def f1_macro_score(y_true, y_pred):
    """Macro f1 score."""
    return f1_score(y_true, y_pred, average='macro')
