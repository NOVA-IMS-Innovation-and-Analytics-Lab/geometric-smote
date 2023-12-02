"""Implementation of the Geometric SMOTE algorithm.

A geometrically enhanced drop-in replacement for SMOTE. It is compatible
with scikit-learn and imbalanced-learn.
"""

from __future__ import annotations

from .geometric_smote import SELECTION_STRATEGIES, GeometricSMOTE, make_geometric_sample

__all__: list[str] = ['GeometricSMOTE', 'make_geometric_sample', 'SELECTION_STRATEGIES']
