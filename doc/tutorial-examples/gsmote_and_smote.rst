.. _gsmote_and_smote:

==================================================
Oversampling visualisation using SMOTE and G-SMOTE
==================================================

.. currentmodule:: gsmote_and_smote

This example illustrates the use of G-SMOTE and SMOTE to
implement an outlier rejections estimator. G-SMOTE can be used easily
within a pipeline. Let's start by importing the necessary libraries::

  >>> from collections import Counter
  >>> from imblearn.pipeline import make_pipeline
  >>> import matplotlib.pyplot as plt
  >>> import numpy as np
  >>>
  >>> # dataset generator
  ... from sklearn.datasets import make_classification
  >>>
  >>> # classifier and oversamplers
  ... from sklearn.svm import LinearSVC
  >>> from imblearn.over_sampling import SMOTE
  >>> from gsmote import GeometricSMOTE

The following function will be used to create a toy dataset. It uses
the :class:`make_classification` from scikit-learn but fixing some
parameters::

  >>> def create_dataset(n_samples=1000, weights=(0.01, 0.01, 0.98),
  ...                    class_sep=0.8, n_clusters=1):
  ...     return make_classification(n_samples=n_samples, n_features=2,
  ...                                n_informative=2, n_redundant=0, n_repeated=0,
  ...                                n_classes=len(weights),
  ...                                n_clusters_per_class=n_clusters,
  ...                                weights=list(weights),
  ...                                class_sep=class_sep, random_state=0)


The following function will be used to plot the sample space after
resampling to illustrate the characteristic of an algorithm.::

  >>> def plot_resampling(X, y, sampler, ax):
  ...     X_res, y_res = sampler.fit_resample(X, y)
  ...     ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
  ...     # make nice plotting
  ...     ax.spines['top'].set_visible(False)
  ...     ax.spines['right'].set_visible(False)
  ...     ax.get_xaxis().tick_bottom()
  ...     ax.get_yaxis().tick_left()
  ...     ax.spines['left'].set_position(('outward', 10))
  ...     ax.spines['bottom'].set_position(('outward', 10))
  ...     return Counter(y_res)

The following function will be used to plot the decision function of a
classifier given some data.::

  >>> def plot_decision_function(X, y, clf, ax):
  ...     plot_step = 0.02
  ...     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  ...     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  ...     xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
  ...                          np.arange(y_min, y_max, plot_step))
  ...     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  ...     Z = Z.reshape(xx.shape)
  ...     ax.contourf(xx, yy, Z, alpha=0.4)
  ...     ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')

:class:`GeometricSMOTE` allows to generate samples. However, this method
of over-sampling does not have any knowledge regarding the underlying
distribution. Therefore, some noisy samples can be generated, e.g. when
the different classes cannot be well separated.::

  >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(15, 25))
  >>>
  >>> X, y = create_dataset(n_samples=1000, weights=(0.1, 0.1, 0.1, 0.7))
  >>>
  >>> ax_arr = ((ax1, ax2), (ax3, ax4))
  >>> for ax, sampler in zip(ax_arr, (
  ...         SMOTE(random_state=None),
  ...         GeometricSMOTE(random_state=None),
  ...         )):
  ...     clf = make_pipeline(sampler, LinearSVC())
  ...     clf.fit(X, y)
  ...     plot_decision_function(X, y, clf, ax[0])
  ...     ax[0].set_title('Decision function for {}'.format(
  ...         sampler.__class__.__name__))
  ...     plot_resampling(X, y, sampler, ax[1])
  ...     ax[1].set_title('Resampling using {}'.format(
  ...         sampler.__class__.__name__))
  ...
  >>> fig.tight_layout()
  >>> plt.show()

**(TODO: Insert generated figures)**
