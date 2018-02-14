"""
Appends new scorers in to the SCORERS constant.
"""

from sklearn.metrics import SCORERS
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from .metrics import (
    weighted_mean_squared_error,
    weighted_mean_absolute_error,
    tp_score,
    tn_score,
    fp_score,
    fn_score)


APPENDED_SCORERS = {
    'weighted_mean_squared_error': (weighted_mean_squared_error, False),
    'weighted_mean_absolute_error': (weighted_mean_absolute_error, False),
    'prediction_coefficient': (weighted_mean_absolute_error, False),
    'tp': (tp_score, ),
    'tn': (tn_score, ),
    'fp': (fp_score, False),
    'fn': (fn_score, False),
    'geometric_mean_score': (geometric_mean_score, )
}

APPENDED_SCORERS = {name:make_scorer(*sc_func) for name, sc_func in APPENDED_SCORERS.items()}

SCORERS.update(APPENDED_SCORERS)
