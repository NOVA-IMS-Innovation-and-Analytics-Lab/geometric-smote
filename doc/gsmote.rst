.. _gsmote:

===============
Geometric SMOTE
===============

.. currentmodule:: gsmote

A practical guide
-----------------

One way to fight the imbalanced learning problem is to generate
new samples in the classes which are under-represented. Douzas
and Bacao (2019) propose Geometric SMOTE: A geometrically enhanced
drop-in replacement for SMOTE [DB2019]_. The
:class:`GeometricSMOTE` is an implementation of the proposed
oversampling strategy. It offers such scheme::

   >>> from collections import Counter
   >>> from sklearn.datasets import make_classification
   >>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
   ...                            n_redundant=0, n_repeated=0, n_classes=3,
   ...                            n_clusters_per_class=1,
   ...                            weights=[0.01, 0.05, 0.94],
   ...                            class_sep=0.8, random_state=0)
   >>> from gsmote import GeometricSMOTE
   >>> gs = GeometricSMOTE(sampling_strategy='auto', random_state=None,
   ...                     truncation_factor=1.0, deformation_factor=0.0,
   ...                     selection_strategy='combined', k_neighbors=5,
   ...                     n_jobs=1
   ...                     )
   >>> X_resampled, y_resampled = gs.fit_resample(X, y)
   >>> from collections import Counter
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 4674), (1, 4674), (2, 4674)]

The augmented data set should be used instead of the original data set
to train a classifier::

   >>> from sklearn.svm import LinearSVC
   >>> clf = LinearSVC()
   >>> clf.fit(X_resampled, y_resampled) # doctest : +ELLIPSIS
   LinearSVC(...)

In the figure below, we compare the decision functions of a classifier
trained using the over-sampled data set and the original data set.

**[TODO]**

.. topic:: References

  .. [DB2019] Douzas, G., & Bacao, F. (2019). Geometric SMOTE a
              geometrically enhanced drop-in replacement for SMOTE.
              Information Sciences, 501, 118–135.
              https://doi.org/10.1016/J.INS.2019.06.007


Sample generation
-----------------

G-SMOTE represents a modification of the original SMOTE algorithm on
the data generation mechanism. Considering a sample :math:`x_i`, a new
sample :math:`x_{new}` will be generated considering its k
nearest-neighbours (corresponding to ``k_neighbors``). Although,
instead of generating an artificial observation within a segment between
:math:`x_i` and one of its k nearest-neighbours (note that all k
observations belong to the minority class), G-SMOTE will randomly generate
the new artificial observation within a geometry, whose shape is determined
by the ``deformation_factor`` and ``truncation_factor``. For more information
the reader is referred to [DB2019]_.


Multi-class management
----------------------

:class:`GeometricSMOTE` can be used with multiple classes as well as
binary classes classification. It requires information regarding the
neighbourhood of each sample used for sample generation.
:class:`GeometricSMOTE` uses a one-vs-rest approach by selecting each
targeted class and computing the necessary statistics against the rest
of the data set which are grouped in a single class.
