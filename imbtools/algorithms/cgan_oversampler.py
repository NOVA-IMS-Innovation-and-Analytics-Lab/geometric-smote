
"""
This module contains the implementation of the 
Conditional Generative Adversarial Network as 
an oversampling algorithm.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
from imblearn.base import BaseBinarySampler
from ganetwork import CGAN, OPTIMIZER
from sklearn.utils import check_random_state


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
        self.cgan_ = CGAN(self.n_Z_features,
                    self.discriminator_hidden_layers,
                    self.generator_hidden_layers,
                    self.discriminator_optimizer,
                    self.discriminator_weights_initilization_choice,
                    self.discriminator_bias_initilization_choice,
                    self.generator_optimizer,
                    self.generator_weights_initilization_choice,
                    self.generator_bias_initilization_choice)
        self.cgan_.train(X, y, self.nb_epoch, self.batch_size, self.discriminator_steps)
        return self

    def _sample(self, X, y):
        random_state = check_random_state(self.random_state)
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.maj_c_] - self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.ratio * self.stats_c_[self.maj_c_] - self.stats_c_[self.min_c_])

        X_resampled = np.concatenate([X, self.cgan_.generate_samples(num_samples, self.min_c_, random_state)], axis=0)
        y_resampled = np.concatenate([y, [self.min_c_] * num_samples], axis=0)

        return X_resampled, y_resampled



    
        
        