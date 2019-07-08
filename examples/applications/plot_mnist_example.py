"""
=========================================
MNIST Dataset classification with G-SMOTE
=========================================
The example makes use of openml's MNIST dataset.
"""

# Authors: Joao Fonseca <jpmrfonseca@gmail.com>
#          Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

import numpy                 as np
import matplotlib.pyplot     as plt
import pandas                as pd
from collections             import Counter
from imblearn.over_sampling  import SMOTE
from imblearn.datasets       import make_imbalance
from gsmote                  import GeometricSMOTE
from sklearn.datasets        import fetch_openml
from collections             import Counter
from random                  import choice

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
# downloaded from the web if necessary. Afterwards fhe function
# :func:`imblearn.datasets.make_imbalance` is used to imbalance the dataset

# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
pre_dist = Counter(y)
X, y = make_imbalance(X=X,y=y, sampling_strategy={
                                        '0':int(pre_dist['0']*.2), '1':int(pre_dist['1']*.18),
                                        '2':int(pre_dist['2']*.16), '3':int(pre_dist['3']*.14),
                                        '4':int(pre_dist['4']*.12), '5':int(pre_dist['5']*.1),
                                        '6':int(pre_dist['6']*.08), '7':int(pre_dist['7']*.06),
                                        '8':int(pre_dist['8']*.04), '9':int(pre_dist['9']*.02)})

dist = Counter(y)
print(pd.DataFrame(dist.values(), index=dist.keys(), columns=['Count']).sort_index())

###############################################################################
# Below is presented a random observation from each class from the original
# dataset

def plot_mnist_samples(X, y, title=None):
    imshape = int(np.sqrt(X.shape[-1]))
    fig, axes = plt.subplots(nrows=1, ncols=np.unique(y).shape[0], figsize=(20, 3))
    if title:
        fig.suptitle(title, fontsize=16)
    for i, val in enumerate(np.unique(y)):
        images = np.where(y==val)[0]
        img = X[choice(images)]
        axes[i].imshow(np.reshape(img, (imshape,imshape)), cmap='gray')
        axes[i].set_title(str(val))
        axes[i].axis('off')



plot_mnist_samples(X, y)

###############################################################################
# Data Generation
###############################################################################

###############################################################################
# Below is presented the generation of new samples using the SMOTE and G-SMOTE
# algorithms. In the case of G-SMOTE

sampling_strategy = {
                 '0':dist['0']+1, '1':dist['1']+1,
                 '2':dist['2']+1, '3':dist['3']+1,
                 '4':dist['4']+1, '5':dist['5']+1,
                 '6':dist['6']+1, '7':dist['7']+1,
                 '8':dist['8']+1, '9':dist['9']+1
                 }

def get_disjoin(X1, y1, X2, y2):
    """returns rows that do not belong to one of the two datasets"""
    if not X1.shape[-1]==X2.shape[-1]:
        raise ValueError('Both arrays must have equal shape on axis 1.')

    if X1.shape[0]>X2.shape[0]: X_largest, y_largest, X_smallest, y_smallest = X1, y1, X2, y2
    else:                       X_largest, y_largest, X_smallest, y_smallest = X2, y2, X1, y1

    shape = (max(X1.shape[0],X2.shape[0]), X1.shape[-1])
    intersecting_vals = np.in1d(X_largest,X_smallest).reshape(shape)
    disjoin_indexes = np.where(~np.all(intersecting_vals, axis=1))[0]
    return X_largest[disjoin_indexes], y_largest[disjoin_indexes]


for strategy in ['combined', 'majority', 'minority']:
    pass

gsmote_sampling = GeometricSMOTE(
                                sampling_strategy=sampling_strategy,
                                n_jobs=-1,
                                selection_strategy='combined').fit_sample(X,y)
X_gsmote_combined, y_gsmote_combined = get_disjoin(X, y, gsmote_sampling[0], gsmote_sampling[1])
plot_mnist_samples(X_gsmote_combined, y_gsmote_combined, 'Data Generated Using G-SMOTE: combined')


gsmote_sampling = GeometricSMOTE(
                                sampling_strategy=sampling_strategy,
                                n_jobs=-1,
                                selection_strategy='majority').fit_sample(X,y)
X_gsmote_majority, y_gsmote_majority = get_disjoin(X, y, gsmote_sampling[0], gsmote_sampling[1])
plot_mnist_samples(X_gsmote_majority, y_gsmote_majority, 'Data Generated Using G-SMOTE: majority')


gsmote_sampling = GeometricSMOTE(
                                sampling_strategy=sampling_strategy,
                                n_jobs=-1,
                                selection_strategy='minority').fit_sample(X,y)
X_gsmote_minority, y_gsmote_minority = get_disjoin(X, y, gsmote_sampling[0], gsmote_sampling[1])
plot_mnist_samples(X_gsmote_minority, y_gsmote_minority, 'Data Generated Using G-SMOTE: minority')


smote_sampling = SMOTE(n_jobs=-1).fit_sample(X,y)
X_smote, y_smote = get_disjoin(X, y, smote_sampling[0], smote_sampling[1])
plot_mnist_samples(X_smote, y_smote, 'Data Generated Using SMOTE')
