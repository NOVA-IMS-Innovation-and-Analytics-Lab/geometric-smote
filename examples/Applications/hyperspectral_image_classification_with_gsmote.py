"""
==============================================
Remote Sensing data classification
==============================================

The example makes use of Indian Pine Test Site 3 (220 Band AVIRIS Hyperspectral 
Image Data Set: June 12, 1992).

"""

# Authors: Manvel Khudinyan <armkhudinyan@gmail.com>
#          Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import datasets 

from imblearn.pipeline import Pipeline
from gsmote import GeometricSMOTE

print(__doc__)


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


###############################################################################
# Indian Pines Test Site 3
###############################################################################

###############################################################################
# This hyperspectral data set has 220 spectral bands and 20 m spatial resolution. 
# The imagery was collected on 12 June 1992 and represents a 2.9 by 2.9 km area 
# in Tippecanoe County, Indiana, USA. The area is agricultural and eight classes 
# as land-use types are presented: alfalfa, corn, grass, hay, oats, soybeans, 
# trees, and wheat. The Indian Pines data set has been used for testing and 
# comparing algorithms. The number of samples varies greatly among the classes, 
# which is known as an imbalanced training set. Data are made available by 
# Purdue University (https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html).

###############################################################################
# Dataset
###############################################################################

###############################################################################
# This dataset provides the data in numpay arrays. Predictor and target 
# variables are already split (X and y accordingly). Predictor data consists 
# of 220 features. Target attributes are the land cover classes. Dataset has 
# 9144 samples.  

# Importing the datad
X, y, feature_names, description = tuple(fetch_openml('Indian_pines').values())[:4]

n_features = (X).shape[1]
n_samples = (X).shape[0]
# Ratio (R)- number of samples over number of features
ratio = n_samples/n_features
print("n_features:'%s', n_samples: '%s', R: '%s'" %
      (n_features, n_samples, ratio))

# Data split into train and test parts
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, 
                                                    test_size=0.33, shuffle=True)

###############################################################################
# Classification
###############################################################################

###############################################################################
# Below we present a simple implementation of Logistic Regression classifier (LR) 
# with Principal Component Analysis (PCA) and G-SMOTE oversampler (combined 
# by pipeline). As the dimensionality of the dataset is too large (220) the PCA 
# analysis become important to reduce the time taken for classification.
# GridSearchCV tool from sklearn is used to find the best parameters for the 
# classifier and oversampler. For model validation 5fold cross-validation is 
# used. F-score measure is implemented to evaluate the classification 
# performance. 

# Implement pipeline
lr_pipeline = Pipeline([
                ('gsmote', GeometricSMOTE(random_state=0)),
                ('pca', PCA(n_components=10)),
                ('lr', LogisticRegression(solver='lbfgs', max_iter=5000)),
])

# Define grid search parameters
param_grid = {
    'gsmote__k_neighbors':[3, 4], 
    'gsmote__deformation_factor': [0.25, 0.50, 0.75], 
    'gsmote__truncation_factor': [-0.5, 0.0, 0.5], 
    'lr__C': [1e3, 1e2, 1e1, 1e0, 1e-1],
}

# Fit the grid search
CV=StratifiedKFold(n_splits=5, shuffle=True) 
grid = GridSearchCV(lr_pipeline, param_grid,  scoring='f1_macro', 
                    cv=CV).fit(X_train, y_train)

print(format(classification_report(grid.best_estimator_.predict(X_test), y_test)))
print(grid.best_score_)
print(grid.best_params_)

###############################################################################
# Classification confusion matrix
###############################################################################

###############################################################################
# To describe the performance of the classification models per classes you can 
# create the normalized confusion matrix. Particularly, this matrix represented 
# the predictive power of the classifier LR with G-SMOTE oversampler in 
# the discrimination of eight classes using 220 Band AVIRIS Hyperspectral Image 
# Data Set (Indian Pine Test Site 3). The values of the diagonal elements 
# represented the degree of correctly predicted classes.

# Plot confusion matrix
cm = confusion_matrix(grid.best_estimator_.predict(X_test), y_test, labels = np.unique(y_test))
np.set_printoptions(precision=2)
plt.figure(figsize = (10,10), dpi=80)
plot_confusion_matrix(cm, classes=np.unique(y_test), normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('confusionmatrix.png', bbox_inches="tight")
plt.show()
print(cm)