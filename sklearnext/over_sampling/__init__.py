"""
The :mod:`sklearnext.oversampling` module includes
 various oversampling methods.
"""

from .cgan_oversampler import CGANOversampler
from .geometric_smote import GeometricSMOTE
from ._oversamplers import SMOTE

__all__ = [
    'CGANOversampler',
    'GeometricSMOTE',
    'SMOTE'
]