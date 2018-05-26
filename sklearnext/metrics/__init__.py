"""
Appends new scorers in to the SCORERS constant.
"""

from sklearn.metrics import SCORERS
from sklearn.metrics import make_scorer
from imblearn.metrics import geometric_mean_score
from .classification import tp_score, tn_score, fp_score, fn_score
from .regression import weighted_mean_squared_error

__all__ = ['geometric_mean_score', 'tp_score', 'tn_score', 'fp_score', 'fn_score', 'SCORERS']

APPENDED_SCORERS = {
    'tp': (tp_score, ),
    'tn': (tn_score, ),
    'fp': (fp_score, False),
    'fn': (fn_score, False),
    'geometric_mean_score': (geometric_mean_score, ),
    'weighted_mean_squared_error': (weighted_mean_squared_error, )
}

APPENDED_SCORERS = {name:make_scorer(*sc_func) for name, sc_func in APPENDED_SCORERS.items()}

SCORERS.update(APPENDED_SCORERS)
