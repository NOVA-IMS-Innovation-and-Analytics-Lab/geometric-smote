"""
MNIST Dataset oversampling
==========================

The example makes use of openml's MNIST dataset.
"""

# Authors: Joao Fonseca <jpmrfonseca@gmail.com>
#          Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from imblearn_extra.gsmote import GeometricSMOTE


def plot_mnist_samples(X, y, title=None, n_subplots=None):
    if n_subplots is None:
        n_subplots = [1, np.unique(y).shape[0]]
    imshape = int(np.sqrt(X.shape[-1]))
    fig, axes = plt.subplots(nrows=n_subplots[0], ncols=n_subplots[1], figsize=(20, 3))
    if title is not None:
        fig.suptitle(title, fontsize=16)
    for i, val in enumerate(np.unique(y)):
        images = X[y == val]
        img = images.iloc[np.random.randint(images.shape[0])]
        if len(np.unique(y)) > 1:
            axes[i].imshow(np.reshape(img.values, (imshape, imshape)), cmap='gray')
            axes[i].set_title(str(val))
            axes[i].axis('off')
        else:
            axes.imshow(np.reshape(img.values, (imshape, imshape)), cmap='gray')
            axes.set_title(str(val))
            axes.axis('off')


def fit_pipelines(X_train, y_train, X_test, y_test):
    pipelines = {
        'No oversampling': Pipeline([('none', None), ('rfc', RandomForestClassifier(random_state=1))]),
        'SMOTE': Pipeline(
            [
                ('smote', SMOTE(random_state=34, k_neighbors=3)),
                ('rfc', RandomForestClassifier(random_state=1)),
            ],
        ),
        'GSMOTE': Pipeline(
            [
                ('gsmote', GeometricSMOTE(random_state=12, k_neighbors=3)),
                ('rfc', RandomForestClassifier(random_state=1)),
            ],
        ),
    }
    results = {}
    for name, estimator in pipelines.items():
        estimator.fit(X_train, y_train)
        results[name] = estimator.score(X_test, y_test)
    return pd.DataFrame(data=results.values(), index=results.keys(), columns=['Score'])


# %%
# MNIST Dataset
# -------------
#
# The MNIST database is composed of handwritten digits with 784 features,
# the raw data is available at: http://yann.lecun.com/exdb/mnist/.
#
# It is a subset of a larger set available from NIST. The digits have been
# size-normalized and centered in a fixed-size image. It is a good database for
# people who want to try learning techniques and pattern recognition methods on
# real-world data while spending minimal efforts on preprocessing and
# formatting. The original black and white (bilevel) images from NIST were size
# normalized to fit in a 20x20 pixel box while preserving their aspect ratio.
# The resulting images contain grey levels as a result of the anti-aliasing
# technique used by the normalization algorithm. the images were centered in a
# 28x28 image by computing the center of mass of the pixels, and translating
# the image so as to position this point at the center of the 28x28 field.
#
# The function `sklearn.datasets.fetch_openml` will load the MNIST
# dataset. It returns a tuple object with the feature matrix as the
# first item and the target values in the second. The dataset will be
# downloaded from the web if necessary. Afterwards we select only 1's and 7's
# from the dataset, create a balanced hold-out set and use the function
# `imblearn.datasets.make_imbalance` to make the dataset imbalanced.

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
selection = y.isin(['1', '7'])
X, y = X[selection], y[selection]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
indices = np.random.RandomState(3).choice(X_train.index, 1000)
X_train, y_train = X_train.loc[indices], y_train.loc[indices]
X_train, y_train = make_imbalance(
    X=X_train,
    y=y_train,
    sampling_strategy={'1': 5},
    random_state=9,
)
X_test, y_test = make_imbalance(
    X=X_test,
    y=y_test,
    sampling_strategy={'1': 100, '7': 100},
    random_state=11,
)
counter_train = Counter(y_train)
counter_test = Counter(y_test)
distribution = pd.DataFrame(
    {'1': [counter_train['1'], counter_test['1']], '7': [counter_train['7'], counter_test['7']]},
).set_axis(['Train', 'Test'])
distribution

# %%
# Below is presented a random observation from each class from the training data:

plot_mnist_samples(X_train, y_train)

# %%
# Data Generation
# ---------------
#
# Below is presented the generation of a new sample using the G-SMOTE
# algorithm for each of the three selection strategies.

for strategy in ['combined', 'majority', 'minority']:
    X_res, y_res = GeometricSMOTE(
        k_neighbors=3,
        selection_strategy=strategy,
        random_state=5,
    ).fit_resample(X_train, y_train)
    plot_mnist_samples(X_res[len(X_train) :], y_res[len(X_train) :], f'Generated Using G-SMOTE: {strategy.title()}')

# %%
# Classification
# --------------
#
# Finally we train a Random Forest Classfier algorithm and optionally use either
# SMOTE and or G-SMOTE oversampling methods to predict the number in each picture of
# this imbalanced dataset. We present the accuracy for each estimator:

fit_pipelines(X_train, y_train, X_test, y_test)
