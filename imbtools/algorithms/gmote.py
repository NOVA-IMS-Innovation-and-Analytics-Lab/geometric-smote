"""
This module contains the implementation of the
Conditional Generative Adversarial Network as
an oversampling algorithm.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.utils import check_random_state
from numpy.linalg import norm


def _generate_sample(center, radius, random_state=None):
    random_state = check_random_state(random_state)
    normal_samples = random_state.normal(size=center.size)
    on_sphere = normal_samples / norm(normal_samples)
    in_ball = (random_state.uniform(size=1) ** (1 / center.size)) * on_sphere
    return center + radius * in_ball

