"""
The :mod:`sklearnext.oversampling.cgan_oversampler`
contains the implementation of the Conditional Generative
Adversarial Network as an oversampling algorithm.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

import numpy as np
from sklearn.utils import check_random_state
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring

from .base import BaseClusterOverSampler


class CGAN:
    pass


@Substitution(
    sampling_strategy=BaseClusterOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class CGANOversampler(BaseClusterOverSampler):
    """Class to perform oversampling using a
    Conditional Generative Adversarial Network as
    an oversampling algorithm.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    clusterer : Clusterer object, optional (default=None)
        A clustering algorithm that is used to generate new samples 
        in each cluster defined by the ``labels_`` attribute and between 
        the clusters if the ``neighbors_`` attribute is defined.

    distributor : Distributor object, optional (default=None)
        Determines the the strategy to distribute generated 
        samples across the clusters.

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
    cgan_ : CGAN object
        A CGAN instance containing the discriminator and generator.
    """

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None, 
                 clusterer=None,
                 distributor=None,
                 n_Z_features=None,
                 discriminator_hidden_layers=None,
                 generator_hidden_layers=None,
                 discriminator_optimizer=None,
                 discriminator_initializer=None,
                 generator_optimizer=None,
                 generator_initializer=None,
                 nb_epoch=None,
                 batch_size=None,
                 discriminator_steps=1):
        super(CGANOversampler, self).__init__(sampling_strategy=sampling_strategy, 
                                              clusterer=clusterer, distributor=distributor)
        self.random_state = random_state
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

        # Define CGAN
        self.cgan_ = CGAN(self.n_Z_features,
                          self.discriminator_hidden_layers,
                          self.generator_hidden_layers,
                          self.discriminator_optimizer,
                          self.discriminator_initializer,
                          self.generator_optimizer,
                          self.generator_initializer)
        # Fit CGAN
        self.cgan_.train(X, y, self.nb_epoch, self.batch_size, self.discriminator_steps)

        return self

    def _basic_sample(self, X, y):
        """Basic resample of the dataset.

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

        # Check random state
        random_state = check_random_state(self.random_state)
        
        # Resample data
        for class_sample, num_samples in self.ratio_.items():
            X_resampled = np.vstack((X, self.cgan_.generate_samples(num_samples, class_sample, random_state)))
            y_resampled = np.hstack((y, [class_sample] * num_samples))

        return X_resampled, y_resampled
