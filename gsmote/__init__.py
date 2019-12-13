"""
The :mod:`gsmote` provides the implementation of
Geometric SMOTE algorithm.
"""

from .geometric_smote import GeometricSMOTE

from ._version import __version__

__all__ = ['GeometricSMOTE', '__version__']
