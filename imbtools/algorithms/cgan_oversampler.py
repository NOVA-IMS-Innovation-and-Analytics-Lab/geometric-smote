
"""
This module contains the implementation of the 
Conditional Generative Adversarial Network as 
an oversampling algorithm.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from imblearn.base import BaseBinarySampler
from ganetwork import CGAN


OPTIMIZER = tf.train.AdamOptimizer()

class CGANOversampler(BaseBinarySampler):
    """Class to perform oversampling using a 
    Conditional Generative Adversarial Network as 
    an oversampling algorithm."""