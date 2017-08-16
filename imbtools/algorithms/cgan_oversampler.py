
"""
This module contains the implementation of the 
Conditional Generative Adversarial Network as 
an oversampling algorithm.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from ganetwork import CGAN, OPTIMIZER
from sklearn.utils import check_random_state


class CGANOversampler(BaseOverSampler):
    """Class to perform oversampling using a 
    Conditional Generative Adversarial Network as 
    an oversampling algorithm.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.
    n_Z_features : int
        Number of features of the Z noise space.
    discriminator_hidden_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the discriminator's corresponding hidden layer.
    generator_hidden_layers : list of (int, activation function) tuples
        Each tuple represents the number of neurons and the activation 
        function of the generators's corresponding hidden layer.
    discriminator_optimizer : TensorFlow optimizer, default AdamOptimizer
        The optimizer for the discriminator.
    generator_optimizer : TensorFlow optimizer, default AdamOptimizer
        The optimizer for the generator.
    discriminator_initializer : list of strings or TensorFlow tensor, default ['xavier', 'zeros']
        The initialization type of the discriminator's parameters.
    generator_initializer : list of strings or TensorFlow tensor, default ['xavier', 'zeros']
        The initialization type of the generator's parameters.
    nb_epoch : int
        Number of epochs for the CGAN training
    batch_size : int
        The minibatch size.
    discriminator_steps : int
        The discriminator update steps followed by a single generator update.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.
    max_c_ : str or int
        The identifier of the majority class.
    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.
    X_shape_ : tuple of int
        Shape of the data `X` during fitting.
    cgan_ : CGAN object
        A CGAN instance containing the discriminator and generator.
    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 n_Z_features=None, 
                 discriminator_hidden_layers=None, 
                 generator_hidden_layers=None, 
                 discriminator_optimizer=OPTIMIZER,  
                 discriminator_initializer=['xavier', 'zeros'],
                 generator_optimizer=OPTIMIZER, 
                 generator_initializer=['xavier', 'zeros'],
                 nb_epoch=None, 
                 batch_size=None, 
                 discriminator_steps=1):
        super().__init__(ratio=ratio, random_state=random_state)
        self.n_Z_features = n_Z_features
        self.discriminator_hidden_layers = discriminator_hidden_layers
        self.generator_hidden_layers = generator_hidden_layers
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_initializer = discriminator_initializer
        self.generator_optimizer = generator_optimizer
        self.generator_initializer = generator_initializer
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.discriminator_steps = discriminator_steps
    
    def fit(self, X, y):
        """Fits CGAN model and finds the classes statistics.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.
        
        Returns
        -------
        self : object,
            Return self.
        """
        super().fit(X, y)
        self.cgan_ = CGAN(self.n_Z_features,
                    self.discriminator_hidden_layers,
                    self.generator_hidden_layers,
                    self.discriminator_optimizer,
                    self.discriminator_initializer,
                    self.generator_optimizer,
                    self.generator_initializer)
        self.cgan_.train(X, y, self.nb_epoch, self.batch_size, self.discriminator_steps, logging_options=None)
        return self

    def _sample(self, X, y):
        """Resample the dataset.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.
        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.
        
        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.
        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`
        """
        random_state = check_random_state(self.random_state)
        for class_sample, num_samples in self.ratio_.items():
            X_resampled = np.concatenate([X, self.cgan_.generate_samples(num_samples, class_sample, random_state)], axis=0)
            y_resampled = np.concatenate([y, [class_sample] * num_samples], axis=0)

        return X_resampled, y_resampled



    
        
        