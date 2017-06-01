
"""
This module contains the implementation of the 
Conditional Generative Adversarial Network as 
an oversampling algorithm.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from imblearn.base import BaseBinarySampler
from ganetwork import CGAN, OPTIMIZER


class CGANOversampler(BaseBinarySampler):
    """Class to perform oversampling using a 
    Conditional Generative Adversarial Network as 
    an oversampling algorithm."""

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 n_Z_features=None, 
                 discriminator_hidden_layers=None, 
                 generator_hidden_layers=None, 
                 discriminator_optimizer=OPTIMIZER,  
                 discriminator_weights_initilization_choice='xavier',
                 discriminator_bias_initilization_choice='zeros',
                 generator_optimizer=OPTIMIZER, 
                 generator_weights_initilization_choice='xavier',
                 generator_bias_initilization_choice='zeros', 
                 nb_epoch=None, 
                 batch_size=None, 
                 discriminator_steps=1):
        super().__init__(ratio=ratio, random_state=random_state)
        self.n_Z_features = n_Z_features
        self.discriminator_hidden_layers = discriminator_hidden_layers
        self.generator_hidden_layers = generator_hidden_layers
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_weights_initilization_choice = discriminator_weights_initilization_choice
        self.discriminator_bias_initilization_choice = discriminator_bias_initilization_choice
        self.generator_optimizer = generator_optimizer
        self.generator_weights_initilization_choice = generator_weights_initilization_choice
        self.generator_bias_initilization_choice = generator_bias_initilization_choice
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
    
    def fit(self, X, y):
        super().fit(X, y)
        cgan = CGAN(self.n_Z_features,
                    self.discriminator_hidden_layers,
                    self.generator_hidden_layers,
                    self.discriminator_optimizer,
                    self.discriminator_weights_initilization_choice,
                    self.discriminator_bias_initilization_choice,
                    self.generator_optimizer,
                    self.generator_weights_initilization_choice,
                    self.generator_bias_initilization_choice)
        cgan.train(X, y, self.nb_epoch, self.batch_size, self.discriminator_steps)
        return self


    
        
        