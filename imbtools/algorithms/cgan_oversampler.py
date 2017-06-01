
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

def __init__(self,
             n_Z_features, 
             discriminator_hidden_layers, 
             generator_hidden_layers, 
             discriminator_optimizer=OPTIMIZER,  
             discriminator_weights_initilization_choice='xavier',
             discriminator_bias_initilization_choice='zeros',
             generator_optimizer=OPTIMIZER, 
             generator_weights_initilization_choice='xavier',
             generator_bias_initilization_choice='zeros', 
             random_state=None):
    