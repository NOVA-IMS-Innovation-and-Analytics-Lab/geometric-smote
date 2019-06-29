.. _gsmote_with_remote_sensing:

====================================================
Land Use/Land Cover maps classification with G-SMOTE
====================================================

.. currentmodule:: gsmote_with_remote_sensing

The examples makes use of scikit-learn's Forest covertypes dataset::

  >>> # utility purposes
  ... from collections import Counter
  >>> import pandas as pd
  >>> from sklearn.model_selection import train_test_split
  >>> from imblearn.metrics import classification_report_imbalanced
  >>>
  >>> # fetch some imbalanced toy data
  ... import sklearn.datasets as datasets
  >>>
  >>> # models, oversampler, pipeline
  ... from imblearn.pipeline import make_pipeline
  >>> from gsmote import GeometricSMOTE
  >>> from sklearn.ensemble import RandomForestClassifier
  >>>
  >>> # Fetch data
  ... X, y, description = datasets.fetch_covtype().values()
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1)
  >>> print(description)

.. _covtype_dataset:

Forest covertypes
-----------------

The samples in this dataset correspond to 30×30m patches of forest in the US,
collected for the task of predicting each patch's cover type,
i.e. the dominant species of tree.
There are seven covertypes, making this a multiclass classification problem.
Each sample has 54 features, described on the
`dataset's homepage <https://archive.ics.uci.edu/ml/datasets/Covertype>`__.
Some of the features are boolean indicators,
while others are discrete or continuous measurements.

**Data Set Characteristics:**

    =================   ============
    Classes                        7
    Samples total             581012
    Dimensionality                54
    Features                     int
    =================   ============

:func:`sklearn.datasets.fetch_covtype` will load the covertype dataset;
it returns a dictionary-like object
with the feature matrix in the ``data`` member
and the target values in ``target``.
The dataset will be downloaded from the web if necessary.

This dataset is clearly imbalanced::
  >>> # Display an overview of observations per class
  ... counts = dict(Counter(y))
  >>> pd.DataFrame(counts.values(), index=counts.keys(), columns=['count']).sort_index()
      count
  1  211840
  2  283301
  3   35754
  4    2747
  5    9493
  6   17367
  7   20510

Below we have a simple implementation of a Random Forest Classifier to predict
the forest type of each patch of forest. Two experiments are ran: One using
only the classifier (without oversampling), another using G-SMOTE (put together
using a pipeline). Afterwards a classification report is shown for both
experiments::

  >>> # Set up experiments using G-SMOTE and no over sampling
  ... clf = RandomForestClassifier(bootstrap=True)
  >>> ovs_clf = make_pipeline(GeometricSMOTE(), RandomForestClassifier(bootstrap=True))
  >>>
  >>> for comb in [clf, ovs_clf]:
  ...     title = f'{comb.__class__.__name__} - Results'
  ...     div = '='*len(title)
  ...     print(div+'\n'+title+'\n'+div)
  ...     comb.fit(X_train, y_train)
  ...     y_pred_bal = comb.predict(X_test)
  ...     print(classification_report_imbalanced(y_test, y_pred_bal))
  ...
  ================================
  RandomForestClassifier - Results
  ================================
  RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                         max_depth=None, max_features='auto', max_leaf_nodes=None,
                         min_impurity_decrease=0.0, min_impurity_split=None,
                         min_samples_leaf=1, min_samples_split=2,
                         min_weight_fraction_leaf=0.0, n_estimators=10,
                         n_jobs=None, oob_score=False, random_state=None,
                         verbose=0, warm_start=False)
                     pre       rec       spe        f1       geo       iba       sup

            1       0.94      0.95      0.97      0.95      0.96      0.92     20926
            2       0.95      0.96      0.95      0.95      0.96      0.91     28518
            3       0.93      0.95      1.00      0.94      0.97      0.95      3573
            4       0.91      0.84      1.00      0.87      0.91      0.82       285
            5       0.93      0.72      1.00      0.81      0.85      0.70       942
            6       0.93      0.85      1.00      0.89      0.92      0.84      1757
            7       0.97      0.93      1.00      0.95      0.96      0.92      2101

  avg / total       0.95      0.95      0.96      0.95      0.95      0.91     58102

  ==================
  Pipeline - Results
  ==================
  Pipeline(memory=None,
           steps=[('geometricsmote',
                   GeometricSMOTE(deformation_factor=0.0, k_neighbors=5, n_jobs=1,
                                  random_state=None, sampling_strategy='auto',
                                  selection_strategy='combined',
                                  truncation_factor=1.0)),
                  ('randomforestclassifier',
                   RandomForestClassifier(bootstrap=True, class_weight=None,
                                          criterion='gini', max_depth=None,
                                          max_features='auto',
                                          max_leaf_nodes=None,
                                          min_impurity_decrease=0.0,
                                          min_impurity_split=None,
                                          min_samples_leaf=1, min_samples_split=2,
                                          min_weight_fraction_leaf=0.0,
                                          n_estimators=10, n_jobs=None,
                                          oob_score=False, random_state=None,
                                          verbose=0, warm_start=False))],
           verbose=False)
                     pre       rec       spe        f1       geo       iba       sup

            1       0.94      0.95      0.97      0.95      0.96      0.92     20926
            2       0.95      0.96      0.95      0.96      0.96      0.91     28518
            3       0.92      0.95      0.99      0.94      0.97      0.94      3573
            4       0.92      0.87      1.00      0.90      0.93      0.86       285
            5       0.93      0.71      1.00      0.81      0.84      0.69       942
            6       0.92      0.84      1.00      0.88      0.92      0.83      1757
            7       0.98      0.93      1.00      0.95      0.96      0.92      2101

  avg / total       0.95      0.95      0.97      0.95      0.96      0.91     58102
