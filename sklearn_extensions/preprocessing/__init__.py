"""
The :mod:`sklearn_extension.preprocessing` module includes
oversampling and feature extraction methods.
"""

from .cgan_oversampler import CGANOversampler
from .geometric_smote import GeometricSMOTE

__all__ = ['CGANOversampler',
           'GeometricSMOTE'
           ]
