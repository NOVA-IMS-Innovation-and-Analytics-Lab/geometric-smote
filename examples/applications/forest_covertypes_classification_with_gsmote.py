"""
====================================================
Land Use/Land Cover maps classification with G-SMOTE
====================================================

The examples makes use of scikit-learn's Forest covertypes dataset.

"""

# Authors: Joao Fonseca <jpmrfonseca@gmail.com>
#          Georgios Douzas <gdouzas@icloud.com>
# Licence: MIT

from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from sklearn.ensemble import RandomForestClassifier
from imblearn.metrics import classification_report_imbalanced
from imblearn.pipeline import make_pipeline

from gsmote import GeometricSMOTE

print(__doc__)


###############################################################################
# Forest covertypes
###############################################################################

###############################################################################
# The samples in this dataset correspond to 30×30m patches of forest in the US,
# collected for the task of predicting each patch's cover type, i.e. the
# dominant species of tree. There are seven covertypes, making this a multiclass
# classification problem. Each sample has 54 features, described on the
# `dataset's homepage <https://archive.ics.uci.edu/ml/datasets/Covertype>`__.
# Some of the features are boolean indicators, while others are discrete or
# continuous measurements.

###############################################################################
# Dataset
###############################################################################

###############################################################################
# The function :func:`sklearn.datasets.fetch_covtype` will load the covertype
# dataset; it returns a dictionary-like object with the feature matrix in the
# ``data`` member and the target values in ``target``. The dataset will be
# downloaded from the web if necessary. This dataset is clearly imbalanced.

X, y, description = datasets.fetch_covtype().values()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1)
print(description)

counts = dict(Counter(y))
print(pd.DataFrame(counts.values(), index=counts.keys(), columns=['count']).sort_index())


###############################################################################
# Classification
###############################################################################

###############################################################################
# Below we have a simple implementation of a Random Forest Classifier to predict
# the forest type of each patch of forest. Two experiments are ran: One using
# only the classifier (without oversampling), another using G-SMOTE (put
# together using a pipeline). Afterwards a classification report is shown for
# both experiments.

clf = RandomForestClassifier(bootstrap=True)
ovs_clf = make_pipeline(GeometricSMOTE(), RandomForestClassifier(bootstrap=True))
for comb in [clf, ovs_clf]:
    title = f'{comb.__class__.__name__} - Results'
    div = '='*len(title)
    print(div+'\n'+title+'\n'+div)
    comb.fit(X_train, y_train)
    y_pred_bal = comb.predict(X_test)
    print(classification_report_imbalanced(y_test, y_pred_bal))
