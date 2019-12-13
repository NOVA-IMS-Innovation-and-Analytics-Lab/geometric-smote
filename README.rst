.. -*- mode: rst -*-

.. _scikit-learn: http://scikit-learn.org/stable/

.. _imbalanced-learn: http://imbalanced-learn.org/en/stable/

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_ |PythonVersion|_ |Pypi|_ |Conda|_ |DOI|_ |Black|_

.. |Travis| image:: https://travis-ci.org/AlgoWit/geometric-smote.svg?branch=master
.. _Travis: https://travis-ci.org/AlgoWit/geometric-smote

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/4u9bgk60o71kmojh/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/georgedouzas/geometric-smote/history

.. |Codecov| image:: https://codecov.io/gh/AlgoWit/geometric-smote/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/AlgoWit/geometric-smote

.. |CircleCI| image:: https://circleci.com/gh/AlgoWit/geometric-smote/tree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/AlgoWit/geometric-smote/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/geometric-smote/badge/?version=latest
.. _ReadTheDocs: https://geometric-smote.readthedocs.io/en/latest/?badge=latest

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/geometric-smote.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/geometric-smote.svg

.. |Pypi| image:: https://badge.fury.io/py/geometric-smote.svg
.. _Pypi: https://badge.fury.io/py/geometric-smote

.. |Conda| image:: https://anaconda.org/algowit/geometric-smote/badges/installer/conda.svg
.. _Conda: https://conda.anaconda.org/algowit

.. |DOI| image:: https://zenodo.org/badge/DOI/10.1016/j.ins.2019.06.007.svg
.. _DOI: https://doi.org/10.1016/j.ins.2019.06.007

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _Black: https://github.com/ambv/black

===============
geometric-smote
===============

Implementation of the Geometric SMOTE algorithm [1]_, a geometrically enhanced
drop-in replacement for SMOTE. It is compatible with scikit-learn_ and
imbalanced-learn_.

Documentation
-------------

Installation documentation, API documentation, and examples can be found on the
documentation_.

.. _documentation: https://geometric-smote.readthedocs.io/en/latest/

Dependencies
------------

geometric-smote is tested to work under Python 3.6+. The dependencies are the
following:

- numpy(>=1.1)
- scikit-learn(>=0.21)
- imbalanced-learn(>=0.4.3)

Additionally, to run the examples, you need matplotlib(>=2.0.0) and
pandas(>=0.22).

Installation
------------

geometric-smote is currently available on the PyPi's repository and you can
install it via `pip`::

  pip install -U geometric-smote

The package is released also in Anaconda Cloud platform::

  conda install -c algowit geometric-smote

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies::

  git clone https://github.com/AlgoWit/geometric-smote.git
  cd geometric-smote
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/AlgoWit/geometric-smote.git

Testing
-------

After installation, you can use `pytest` to run the test suite::

  make test

About
-----

If you use geometric-smote in a scientific publication, we would appreciate
citations to the following paper::

  @article{Douzas2019,
    doi = {10.1016/j.ins.2019.06.007},
    url = {https://doi.org/10.1016/j.ins.2019.06.007},
    year = {2019},
    month = oct,
    publisher = {Elsevier {BV}},
    volume = {501},
    pages = {118--135},
    author = {Georgios Douzas and Fernando Bacao},
    title = {Geometric {SMOTE} a geometrically enhanced drop-in replacement for {SMOTE}},
    journal = {Information Sciences}
  }

Classification of imbalanced datasets is a challenging task for standard
algorithms. Although many methods exist to address this problem in different
ways, generating artificial data for the minority class is a more general
approach compared to algorithmic modifications. SMOTE algorithm [2]_, as well
as any other oversampling method based on the SMOTE mechanism, generates
synthetic samples along line segments that join minority class instances.
Geometric SMOTE (G-SMOTE) is an enhancement of the SMOTE data generation
mechanism. G-SMOTE generates synthetic samples in a geometric region of the
input space, around each selected minority instance.

References:
-----------

.. [1] G. Douzas, F. Bacao, "Geometric SMOTE:
   a geometrically enhanced drop-in replacement for SMOTE",
   Information Sciences, vol. 501, pp. 118-135, 2019.

.. [2] N. V. Chawla, K. W. Bowyer, L. O. Hall, W. P. Kegelmeyer, "SMOTE:
   synthetic minority over-sampling technique", Journal of Artificial
   Intelligence Research, vol. 16, pp. 321-357, 2002.
