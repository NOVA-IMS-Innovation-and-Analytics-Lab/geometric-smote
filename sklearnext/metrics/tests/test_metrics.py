"""
Test the classification and regression modules.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import numpy as np
from ..classification import tp_score, tn_score, fp_score, fn_score

Y_TRUE = np.array([1, 1, 0, 0, 1, 1])
Y_PRED = np.array([1, 0, 1, 0, 1, 1])


def test_tp_score():
    assert tp_score(Y_TRUE, Y_PRED) == 3


def test_tn_score():
    assert tn_score(Y_TRUE, Y_PRED) == 1


def test_fp_score():
    assert fp_score(Y_TRUE, Y_PRED) == 1


def test_fn_score():
    assert fn_score(Y_TRUE, Y_PRED) == 1