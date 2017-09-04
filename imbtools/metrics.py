"""
This module contains various metrics for imbalanced classification tasks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from sklearn.metrics import SCORERS
from sklearn.metrics import make_scorer, confusion_matrix
from imblearn.metrics import geometric_mean_score

def tp_score(y_true, y_pred):
    """True positive score function."""
    return confusion_matrix(y_true, y_pred)[0, 0]

def tn_score(y_true, y_pred):
    """True negative score function."""
    return confusion_matrix(y_true, y_pred)[1, 1]

def fp_score(y_true, y_pred):
    """False positive score function."""
    return confusion_matrix(y_true, y_pred)[1, 0]

def fn_score(y_true, y_pred):
    """False negative score function."""
    return confusion_matrix(y_true, y_pred)[0, 1]

CONF_MATRIX_MAP = [('tp', tp_score, True), ('tn', tn_score, True), ('fp', fp_score, False), ('fn', fn_score, False)]

for score_name, score, greater_is_better in CONF_MATRIX_MAP:
    SCORERS[score_name] = make_scorer(score, greater_is_better)
SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
