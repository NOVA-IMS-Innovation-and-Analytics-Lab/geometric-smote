"""
This module contains various metrics for imbalanced classification tasks.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from sklearn.metrics import SCORERS
from sklearn.metrics import make_scorer, confusion_matrix
from imblearn.metrics import geometric_mean_score

def tp_score(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]
def tn_score(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]
def fp_score(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]
def fn_score(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]
SCORERS['tp'] = make_scorer(tp_score)
SCORERS['tn'] = make_scorer(tn_score)
SCORERS['fp'] = make_scorer(fp_score, greater_is_better=False)
SCORERS['fn'] = make_scorer(fn_score, greater_is_better=False)
SCORERS['geometric_mean_score'] = make_scorer(geometric_mean_score)
