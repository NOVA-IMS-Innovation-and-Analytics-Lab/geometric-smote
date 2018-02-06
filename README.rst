hyperparam-learn
================

Easy and efficient search for optimal hyperparameters and pipelines of estimators in scikit-learn. 
It includes a class to simplify the creation of the hyperparameter space.

|Travis|_ |AppVeyor|_ |Coveralls|_ |CircleCI|_ |Python35|_

.. |Travis| image:: https://travis-ci.org/gdouzas/stack-learn.svg?branch=master
.. _Travis: https://travis-ci.org/gdouzas/stack-learn

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/ppd9qtsk3y8bpi3s?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/gdouzas/stack-learn/history

.. |Coveralls| image:: https://coveralls.io/repos/github/gdouzas/stack-learn/badge.svg?branch=master
.. _Coveralls: https://coveralls.io/github/gdouzas/stack-learn?branch=master

.. |CircleCI| image:: https://circleci.com/gh/gdouzas/stack-learn.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/gdouzas/stack-learn/tree/master

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/scikit-learn

stacklearn
==========

A package for generalized model stacking. The stacklearn package implements the Super Learner, Subsemble algorithms
as well as various experimental metalearning algorithms.

imbalanced-tools
================

A collection of tools for the imbalanced learning problem. It includes classes to simplify the comparison and evaluation
of oversampling algorithms, functions to export the results and various ovarsampling algorithms which are not included
in imbalanced-learn.

metric-learn
============

A collection of classification and regression metrics for scikit-learn.

oversampling-learn
==================

Classification and regression oversamplers which are not included in imbalanced-learn.

References
==========

1. `Stacked Regressions (1996) <http://link.springer.com/article/10.1007/BF00117832>`_
, Leo Breiman. Machine Learning Journal.

2. `Combining Estimates in Regression and Classification (1996) <http://www.tandfonline.com/doi/abs/10.1080/01621459.1996.10476733>`_
, Michael Leblanc, Rob Tibshirani. Journal of the American Statistical Association.

3. `Super Learner (2007) <http://biostats.bepress.com/ucbbiostat/paper222>`_
, Mark van der Laan, Eric Polley, Alan Hubbard. U.C. Berkeley Division of Biostatistics Working Paper Series.

4. `Subsemble: An ensemble method for combining subset-specific algorithm fits (2013) <https://www.ncbi.nlm.nih.gov/pubmed/24778462>`_
, Stephanie Sapp, Mark van der Laan, John Canny. Journal of Applied Statistics.

Prerequisites
=============
- Pandas
- Scikit-Learn
- Imbalanced-Learn

Installation
============

.. code:: shell

    git clone https://github.com/gdouzas/hyperparam-learn
    cd hyperparam-learn
    python3 setup.py install