"""
=======================
Remote sensing examples
=======================

The following examples make use of scikit-learn's Forest Cover Type dataset and
the Indian Pines dataset.

"""

# Authors: Joao Fonseca <jpmrfonseca@gmail.com>
#          Manvel Khudinyan <armkhudinyan@gmail.com>
#          Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from collections import Counter
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml, fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, make_scorer, cohen_kappa_score
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline, Pipeline

from gsmote import GeometricSMOTE

print(__doc__)

RANDOM_STATE = 5


def print_class_counts(y):
    """Print the class counts."""
    counts = dict(Counter(y))
    class_counts = pd.DataFrame(counts.values(), index=counts.keys(), columns=['Count']).sort_index()
    print(class_counts)


def print_classification_report(clf, X_train, X_test, y_train, y_test):
    """Fit classifier and print classification report."""
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    clf_name = clf.__class__.__name__
    div = '=' * len(clf_name)
    title = f'\n{div}\n{clf_name}\n{div}\n'
    print(title, classification_report_imbalanced(y_test, y_pred))


def plot_confusion_matrix(cm, classes):
    """This function prints and plots the 
    normalized confusion matrix."""
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


###############################################################################
# Forest Cover Type
###############################################################################

###############################################################################
# The samples in this dataset correspond to 30×30m patches of forest in the US,
# collected for the task of predicting each patch's cover type, i.e. the
# dominant species of tree. There are seven covertypes, making this a multiclass
# classification problem. Each sample has 54 features, described on the
# `dataset's homepage <https://archive.ics.uci.edu/ml/datasets/Covertype>`_.
# Some of the features are boolean indicators, while others are discrete or
# continuous measurements.

###############################################################################
# Dataset
#..............................................................................
#
# The function :func:`sklearn.datasets.fetch_covtype` will load dataset. It will
# be downloaded from the web if necessary. This dataset is clearly imbalanced.

X, y = fetch_covtype(return_X_y=True)
print_class_counts(y)

###############################################################################
# Classification
#..............................................................................
#
# Below we use the Random Forest Classifier to predict the forest type of each
# patch of forest. Two experiments are ran: One using only the classifier and
# another that creates a pipeline of Geometric SMOTE and the classifier. A
# classification report is printed for both experiments.

splitted_data = train_test_split(X, y, test_size=0.95, random_state=RANDOM_STATE, shuffle=True)

clf = RandomForestClassifier(bootstrap=True, n_estimators=10, random_state=RANDOM_STATE)
ovs_clf = make_pipeline(GeometricSMOTE(random_state=RANDOM_STATE), clf)

print_classification_report(clf, *splitted_data)
print_classification_report(ovs_clf, *splitted_data)

###############################################################################
# Indian Pines
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
#..............................................................................
#
# This dataset provides the data in numpay arrays. Predictor and target 
# variables are already split (X and y accordingly). Predictor data consists 
# of 220 features. Target attributes are the land cover classes. Dataset has 
# 9144 samples.  

X, y, *_ = fetch_openml('Indian_pines').values()
print_class_counts(y)

###############################################################################
# Classification 
#..............................................................................
#
# Below we use the Geometric SMOTE oversampler and Decision Tree classifier,
# combined by a pipeline. GridSearchCV class from scikit-learn is used to find the
# best parameters of the oversampler.

splitted_data = train_test_split(X, y, test_size=0.5, random_state=RANDOM_STATE, shuffle=True)

param_grid = {
    'gsmote__deformation_factor': [0.25, 0.50, 0.75], 
    'gsmote__truncation_factor': [-0.5, 0.0, 0.5]
}
clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
ovs_clf = Pipeline([
    ('gsmote', GeometricSMOTE(random_state=RANDOM_STATE)),
    ('dt', DecisionTreeClassifier(random_state=RANDOM_STATE)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = make_scorer(cohen_kappa_score)
gscv = GridSearchCV(ovs_clf, param_grid, scoring=scoring, refit=True, cv=cv, n_jobs=-1)

print_classification_report(clf, *splitted_data)
print_classification_report(gscv, *splitted_data)

###############################################################################
# Confusion matrix
#..............................................................................
#
# To describe the performance of the classification models per classes you can 
# create the normalized confusion matrix. Particularly, this matrix represented 
# the predictive power of the classifier LR with G-SMOTE oversampler in 
# the discrimination of eight classes using 220 Band AVIRIS Hyperspectral Image 
# Data Set (Indian Pine Test Site 3). The values of the diagonal elements 
# represented the degree of correctly predicted classes.

_, X_test, _, y_test = splitted_data
conf_matrix = confusion_matrix(y_test, gscv.predict(X_test), labels = np.unique(y_test))
plot_confusion_matrix(conf_matrix, classes=np.unique(y_test))
