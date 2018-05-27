"""
Test the classification module.
"""

from ...metrics import tp_score, tn_score, fp_score, fn_score

Y_TRUE = [1, 1, 0, 0, 1, 1]
Y_PRED = [1, 0, 1, 0, 1, 1]


def test_scores():
    assert tp_score(Y_TRUE, Y_PRED) == 3
    assert tn_score(Y_TRUE, Y_PRED) == 1
    assert fp_score(Y_TRUE, Y_PRED) == 1
    assert fn_score(Y_TRUE, Y_PRED) == 1