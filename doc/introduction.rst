.. _introduction:

============
Introduction
============

.. _api_gsmote:

G-SMOTE API
-----------

GeometricSMOTE follows the scikit-learn API using the base estimator
and adding a sampling functionality through the ``sample`` method:

:Estimator:

    The base object, implements a ``fit`` method to learn from data, either::

      estimator = obj.fit(data, targets)

:Resampler:

    To resample a data sets, each sampler implements::

      data_resampled, targets_resampled = obj.fit_resample(data, targets)

Imbalanced-learn samplers accept the same inputs that in scikit-learn:

* ``data``: array-like (2-D list, pandas.Dataframe, numpy.array) or sparse
  matrices;
* ``targets``: array-like (1-D list, pandas.Series, numpy.array).


Problem statement regarding imbalanced data sets
------------------------------------------------

The learning phase and the subsequent prediction of machine learning algorithms
can be affected by the problem of imbalanced data set. The balancing issue
corresponds to the difference of the number of samples in the different
classes. With a greater imbalance ratio, the decision function favours the class
with the larger number of samples, usually referred as the majority class. For 
a visual representation, the reader is referred to
:ref:`https://imbalanced-learn.readthedocs.io/en/stable/introduction.html#problem-statement-regarding-imbalanced-data-sets`.
