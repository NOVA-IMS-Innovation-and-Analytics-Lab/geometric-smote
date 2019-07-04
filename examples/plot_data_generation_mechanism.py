"""
=========================
Data generation mechanism
=========================

This example illustrates the Geometric SMOTE data 
generation mechanism and the usage of its 
hyperparameters.

"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from imblearn.over_sampling import SMOTE

from gsmote import GeometricSMOTE

print(__doc__)


def plot_scatter(X, y, title):
    """Function to plot some data as a scatter plot."""
    plt.figure()
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Positive Class')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Negative Class')
    plt.xlim(-3.0, 3.0)
    plt.ylim(0.0, 4.0)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(title)


###############################################################################
# Simulated imbalanced data
###############################################################################

###############################################################################
# We are generating a highly imbalanced non Gaussian data set. Only two samples
# from the minority (positive) class are included to illustrate the Geometric
# SMOTE data generation mechanism.

X_train_neg, _ = make_blobs(n_samples=100, centers=[(-2.0, 2.25), (1.0, 2.0)], 
                            cluster_std=0.25, random_state=0)
X_train_pos = np.array([[-0.5, 2.3], [-0.3, 2.7]])
X_train = np.vstack([X_train_neg, X_train_pos])
y_train = np.hstack(
    [np.zeros(X_train_neg.shape[0], dtype=np.int8), np.ones(2, dtype=np.int8)]
)
plot_scatter(X_train, y_train, 'Training data')

###############################################################################
# Geometric hyperparameters
###############################################################################

###############################################################################
# Three important hyperparameters of Geometric SMOTE algorithm are
# ``truncation_factor``, ``deformation_factor`` and ``selection_strategy``. They
# are called geometric hyperparameters and they control the characteristics of
# the data generation process allowing to generate diverse synthetic data.

###############################################################################
# Hypersphere
#..............................................................................
#
# Selecting as values of geometric hyperparameters ``truncation_factor=0.0``,
# ``deformation_factor=0.0`` and ``selection_strategy='minority'`` the data
# generation area corresponds to a circle with center as one of the two minority
# class samples and radius equal to the distance between them. In the
# multi-dimensional case the corresponding area is a hypersphere. Therefore the
# plot below shows the superposition of two cyclic areas.

gsmote = GeometricSMOTE(k_neighbors=1, truncation_factor=0.0,
                        deformation_factor=0.0, selection_strategy='minority', 
                        random_state=0)
X_resampled, y_resampled = gsmote.fit_resample(X_train, y_train)
plot_scatter(X_resampled, y_resampled, 'Resampled data')

###############################################################################
# Majority hypersphere
#..............................................................................
#
# Selecting as values of geometric hyperparameters ``truncation_factor=0.0``,
# ``deformation_factor=0.0`` and ``selection_strategy='majority'`` the data
# generation area corresponds to a circle with center as one of the two minority
# class samples and radius equal to the distance between its closest majority
# class neighbor.

gsmote = GeometricSMOTE(k_neighbors=1, truncation_factor=0.0,
                        deformation_factor=0.0, selection_strategy='majority', 
                        random_state=0)
X_resampled, y_resampled = gsmote.fit_resample(X_train, y_train)
plot_scatter(X_resampled, y_resampled, 'Resampled data')

###############################################################################
# Half-hypersphere
#..............................................................................
#
# Selecting as values of geometric hyperparameters ``truncation_factor=1.0``,
# ``deformation_factor=0.0`` and ``selection_strategy='minority'`` the data
# generation area corresponds to a half-circle with center as one of the two
# minority class samples and radius equal to the distance between them. The
# truncated part is defined by the vector that starts from the initially
# selected minority class sample and points to one of its nearestes neighbors.

gsmote = GeometricSMOTE(k_neighbors=1, truncation_factor=1.0,
                        deformation_factor=0.0, selection_strategy='minority', 
                        random_state=0)
X_resampled, y_resampled = gsmote.fit_resample(X_train, y_train)
plot_scatter(X_resampled, y_resampled, 'Resampled data')

###############################################################################
# Hyperellipsis
#..............................................................................
#
# Selecting as values of geometric hyperparameters ``truncation_factor=0.0``,
# ``deformation_factor=0.5`` and ``selection_strategy='minority'`` the data
# generation area deforms to an ellipsis. Again in the multi-dimensional case
# the corresponding area is a hyperellipsis. Therefore the plot below shows the
# superposition of two elliptic areas.

gsmote = GeometricSMOTE(k_neighbors=1, truncation_factor=0.0,
                        deformation_factor=0.5, selection_strategy='minority', 
                        random_state=0)
X_resampled, y_resampled = gsmote.fit_resample(X_train, y_train)
plot_scatter(X_resampled, y_resampled, 'Resampled data')

###############################################################################
# Line segment
#..............................................................................
#
# Selecting as values of geometric hyperparameters ``truncation_factor=0.0``,
# ``deformation_factor=1.0`` and ``selection_strategy='minority'`` the
# deformation of the data generation area increases and becomes a line segment.

gsmote = GeometricSMOTE(k_neighbors=1, truncation_factor=0.0,
                        deformation_factor=1.0, selection_strategy='minority', 
                        random_state=0)
X_resampled, y_resampled = gsmote.fit_resample(X_train, y_train)
plot_scatter(X_resampled, y_resampled, 'Resampled data')

###############################################################################
# SMOTE
#..............................................................................
#
# Selecting as values of geometric hyperparameters ``truncation_factor=1.0``,
# ``deformation_factor=1.0`` and ``selection_strategy='minority'``, we get the
# SMOTE data generation mechanism.

gsmote = GeometricSMOTE(k_neighbors=1, truncation_factor=1.0,
                        deformation_factor=1.0, selection_strategy='minority', 
                        random_state=0)
X_resampled, y_resampled = gsmote.fit_resample(X_train, y_train)
plot_scatter(X_resampled, y_resampled, 'Resampled data')

###############################################################################
# Avoiding the generation of noisy samples
###############################################################################

###############################################################################
# We are adding a third minority class sample to illustrate the difference
# between SMOTE and Geometric SMOTE data generation mechanisms.

X_train_pos = np.array([[-0.5, 2.3], [-0.3, 2.7]])
X_train = np.vstack([X_train, np.array([2.0, 2.0])])
y_train = np.hstack([y_train, np.ones(1, dtype=np.int8)])
plot_scatter(X_train, y_train, 'Training data')

###############################################################################
# SMOTE
#..............................................................................
#
# Increasing the number of ``k_neighbors`` results to the generation of noisy
# samples.

smote = SMOTE(k_neighbors=2, random_state=0)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
plot_scatter(X_resampled, y_resampled, 'Resampled data')

###############################################################################
# Geometric SMOTE
#..............................................................................
#
# Geometric SMOTE on the other hand is protected when the ``selection_strategy``
# values are either ``combined`` or ``majority``. 

gsmote = GeometricSMOTE(k_neighbors=2, truncation_factor=0.0,
                        deformation_factor=0.0, selection_strategy='combined', 
                        random_state=0)
X_resampled, y_resampled = gsmote.fit_resample(X_train, y_train)
plot_scatter(X_resampled, y_resampled, 'Resampled data')



