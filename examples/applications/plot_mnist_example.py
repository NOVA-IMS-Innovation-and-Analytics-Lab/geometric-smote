"""
==========================
MNIST Dataset oversampling
==========================

The example makes use of openml's MNIST dataset.

"""

# Authors: Joao Fonseca <jpmrfonseca@gmail.com>
#          Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from random import choice
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, train_test_split, PredefinedSplit
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from imblearn.datasets import make_imbalance
from imblearn.pipeline import Pipeline

from gsmote import GeometricSMOTE

###############################################################################
# MNIST Dataset
###############################################################################

###############################################################################
# The MNIST database is composed of handwritten digits with 784 features,
# the raw data is available at: http://yann.lecun.com/exdb/mnist/.

###############################################################################
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

###############################################################################
# The function :func:`sklearn.datasets.fetch_openml` will load the MNIST
# dataset; it returns a tuple object with the feature matrix as the
# first item and the target values in the second. The dataset will be
# downloaded from the web if necessary. Afterwards we select only 1's and 7's
# from the dataset, create a balanced hold-out set and use function
# :func:`imblearn.datasets.make_imbalance` imbalance the dataset.

_X, _y = fetch_openml("mnist_784", version=1, return_X_y=True)
selection = _y.isin(["1", "7"])
X = _X[selection]
y = _y[selection]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
X_train, y_train = make_imbalance(
    X=X_train,
    y=y_train,
    sampling_strategy={"1": 2, "7": int(Counter(y_train)["7"] * 0.02)},
)


for t, y_ in [("Train", y_train), ("Test", y_test)]:
    dist = Counter(y_)
    title = f"{t.title()} set data distribution"
    sep = "=" * len(title)
    print(f"{sep}\n{title}\n{sep}")
    print(
        pd.DataFrame(dist.values(), index=dist.keys(), columns=["Count"]).sort_index()
    )

###############################################################################
# Below is presented a random observation from each class from this dataset:


def plot_mnist_samples(X, y, title=None, n_subplots=None):
    if not n_subplots:
        n_subplots = [1, np.unique(y).shape[0]]
    imshape = int(np.sqrt(X.shape[-1]))
    fig, axes = plt.subplots(nrows=n_subplots[0], ncols=n_subplots[1], figsize=(20, 3))
    if title:
        fig.suptitle(title, fontsize=16)
    for i, val in enumerate(np.unique(y)):
        images = X[y == val]
        img = images.iloc[np.random.randint(images.shape[0])]
        if len(np.unique(y)) > 1:
            axes[i].imshow(np.reshape(img.values, (imshape, imshape)), cmap="gray")
            axes[i].set_title(str(val))
            axes[i].axis("off")
        else:
            axes.imshow(np.reshape(img.values, (imshape, imshape)), cmap="gray")
            axes.set_title(str(val))
            axes.axis("off")


plot_mnist_samples(X_train, y_train)

###############################################################################
# Data Generation
###############################################################################

###############################################################################
# Below is presented the generation of new samples using the G-SMOTE
# algorithm. The parameters `selection_strategy`, `deformation_factor` (d)
# and `truncation_factor` (t) vary.


def get_disjoin(X1, y1, X2, y2):
    """returns rows that do not belong to one of the two datasets"""
    if X1.shape[-1] != X2.shape[-1]:
        raise ValueError("Both arrays must have equal shape on axis 1.")

    if X1.shape[0] > X2.shape[0]:
        X_largest, y_largest, X_smallest, y_smallest = X1, y1, X2, y2
    else:
        X_largest, y_largest, X_smallest, y_smallest = X2, y2, X1, y1

    intersecting_vals = np.in1d(X_largest, X_smallest).reshape(X_largest.shape)
    disjoin_indexes = np.where(~np.all(intersecting_vals, axis=1))[0]
    return X_largest.iloc[disjoin_indexes], y_largest.iloc[disjoin_indexes]


for strategy in ["combined", "majority", "minority"]:
    X_gsmote_final = np.empty(shape=(0, X_train.shape[-1]))
    y_gsmote_final = np.empty(shape=(0))
    for d in [0, 0.5, 1]:
        for t in [-1, 0, 1]:
            gsmote_sampling = GeometricSMOTE(
                k_neighbors=1,
                deformation_factor=d,
                truncation_factor=t,
                n_jobs=-1,
                selection_strategy=strategy,
            ).fit_resample(X_train, y_train)
            X_gsmote, _ = get_disjoin(
                X_train, y_train, gsmote_sampling[0], gsmote_sampling[1]
            )
            X_gsmote_final = np.append(X_gsmote_final, X_gsmote, axis=0)
            y_gsmote_final = np.append(
                y_gsmote_final, np.array([f"t={t}, d={d}"] * X_gsmote.shape[0]), axis=0
            )
    plot_mnist_samples(
        pd.DataFrame(X_gsmote_final),
        pd.Series(y_gsmote_final),
        f"Generated Using G-SMOTE: {strategy}",
    )

###############################################################################
# Below is presented the generation of new samples using the SMOTE
# algorithm. Since there is only two instances with the label '1', `k_neighbors`
# is fixed to 1

smote_sampling = SMOTE(
    k_neighbors=1,
    n_jobs=-1,
).fit_resample(X_train, y_train)
X_smote, _ = get_disjoin(X_train, y_train, smote_sampling[0], smote_sampling[1])
X_smote_final = X_smote[:10]
y_smote_final = np.array([f"Sample {n}" for n in range(10)])

plot_mnist_samples(
    X_smote_final, y_smote_final, f"Generated Using SMOTE, K neighbors: 1"
)

###############################################################################
# Classification
###############################################################################

###############################################################################
# Finally we train a Logistic Regression algorithm using the SMOTE and G-SMOTE
# oversamling methods to predict the number in each picture of this imbalanced
# (binary) dataset. A total of 3 pipelines are fit:
# 3 (SMOTE, G-SMOTE, No Oversampling) * 1 (LogisticRegression).


def model_fit(X_train, y_train, X_test, y_test):
    classifier_dict = {
        "no_oversampling": Pipeline(
            [("none", None), ("lr", LogisticRegression(solver="liblinear"))]
        ),
        "smote": Pipeline(
            [
                ("smote", SMOTE(k_neighbors=1)),
                ("lr", LogisticRegression(solver="liblinear")),
            ]
        ),
        "gsmote": Pipeline(
            [
                ("gsmote", GeometricSMOTE(k_neighbors=1)),
                ("lr", LogisticRegression(solver="liblinear")),
            ]
        ),
    }
    results = {}
    for name, estimator in classifier_dict.items():
        estimator.fit(X_train, y_train)
        results[name] = estimator.score(X_test, y_test)
    return pd.DataFrame(data=results.values(), index=results.keys(), columns=["score"])


results = model_fit(X_train, y_train, X_test, y_test)
print(results)
