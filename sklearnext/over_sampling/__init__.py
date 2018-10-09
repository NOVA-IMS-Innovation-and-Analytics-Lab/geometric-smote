"""
The :mod:`sklearnext.oversampling` module includes
 various oversmapling methods.
"""

from .cgan_oversampler import CGANOversampler
from .geometric_smote import GeometricSMOTE

__all__ = [
    'CGANOversampler',
    'GeometricSMOTE'
]