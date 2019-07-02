###########################
Installation & Contribution
###########################

Prerequisites
=============

The geometric-smote package requires the following dependencies:

* numpy (>=1.11)
* scipy (>=0.17)
* scikit-learn (>=0.21)
* imbalanced-learn (>=0.4.3)

Install
=======

geometric-smote is currently available on the PyPi's reporitories and you can
install it via `pip`::

  pip install -U geometric-smote

The package is release also in Anaconda Cloud platform::

  conda install -c conda-forge geometric-smote

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from Github and install all dependencies::

  git clone https://github.com/IMS-ML-Lab/geometric-smote.git
  cd geometric-smote
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/IMS-ML-Lab/geometric-smote.git

Test and coverage
=================

You want to test the code before to install::

  $ make test

You wish to test the coverage of your version::

  $ make coverage

You can also use `pytest`::

  $ pytest gsmote -v

Contribute
==========

You can contribute to this code through Pull Request on GitHub_. Please, make
sure that your code is coming with unit tests to ensure full coverage and
continuous integration in the API.

.. _GitHub: https://github.com/IMS-ML-Lab/geometric-smote/pulls
