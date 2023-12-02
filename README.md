[black badge]: <https://img.shields.io/badge/%20style-black-000000.svg>
[black]: <https://github.com/psf/black>
[docformatter badge]: <https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg>
[docformatter]: <https://github.com/PyCQA/docformatter>
[ruff badge]: <https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json>
[ruff]: <https://github.com/charliermarsh/ruff>
[mypy badge]: <http://www.mypy-lang.org/static/mypy_badge.svg>
[mypy]: <http://mypy-lang.org>
[mkdocs badge]: <https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat>
[mkdocs]: <https://squidfunk.github.io/mkdocs-material>
[version badge]: <https://img.shields.io/pypi/v/geometric-smote.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/geometric-smote.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/geometric-smote>
[gitter]: <https://gitter.im/geometric-smote/community>
[gitter badge]: <https://badges.gitter.im/join%20chat.svg>
[discussions]: <https://github.com/georgedouzas/geometric-smote/discussions>
[discussions badge]: <https://img.shields.io/github/discussions/georgedouzas/geometric-smote>
[ci]: <https://github.com/georgedouzas/geometric-smote/actions?query=workflow>
[ci badge]: <https://github.com/georgedouzas/geometric-smote/actions/workflows/ci.yml/badge.svg?branch=main>
[doc]: <https://github.com/georgedouzas/geometric-smote/actions?query=workflow>
[doc badge]: <https://github.com/georgedouzas/geometric-smote/actions/workflows/doc.yml/badge.svg?branch=main>

# geometric-smote

[![ci][ci badge]][ci] [![doc][doc badge]][doc]

| Category          | Tools    |
| ------------------| -------- |
| **Development**   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![mkdocs][mkdocs badge]][mkdocs]|
| **Communication** | [![gitter][gitter badge]][gitter] [![discussions][discussions badge]][discussions] |

## Introduction

The package `geometric-smote` implements the Geometric SMOTE algorithm, a geometrically enhanced drop-in replacement for SMOTE. It
is compatible with scikit-learn and imbalanced-learn. The Geometric SMOTE algorithm can handle numerical as well as categorical
features.

## Installation

For user installation, `geometric-smote` is currently available on the PyPi's repository, and you can
install it via `pip`:

```bash
pip install geometric-smote
```

Development installation requires cloning the repository and then using [PDM](https://github.com/pdm-project/pdm) to install the
project as well as the main and development dependencies:

```bash
git clone https://github.com/georgedouzas/geometric-smote.git
cd geometric-smote
pdm install
```

## Usage

All the classes included in `geometric-smote` follow the [imbalanced-learn] API using the functionality of the base
oversampler. Using [scikit-learn] convention, the data are represented as follows:

- Input data `X`: 2D array-like or sparse matrices.
- Targets `y`: 1D array-like.

The clustering-based oversamplers implement a `fit` method to learn from `X` and `y`:

```python
gsmote_oversampler.fit(X, y)
```

They also implement a `fit_resample` method to resample `X` and `y`:

```python
X_resampled, y_resampled = gsmote.fit_resample(X, y)
```

## Citing `geometric-smote`

If you use `geometric-smote` in a scientific publication, we would appreciate citations to the following paper:

- Douzas, G., Bacao, B. (2019). Geometric SMOTE: a geometrically enhanced
  drop-in replacement for SMOTE. Information Sciences, 501, 118-135.
  <https://doi.org/10.1016/j.ins.2019.06.007>

Publications using Geometric-SMOTE:

- Fonseca, J., Douzas, G., Bacao, F. (2021). Increasing the Effectiveness of
  Active Learning: Introducing Artificial Data Generation in Active Learning
  for Land Use/Land Cover Classification. Remote Sensing, 13(13), 2619.
  <https://doi.org/10.3390/rs13132619>

- Douzas, G., Bacao, F., Fonseca, J., Khudinyan, M. (2019). Imbalanced
  Learning in Land Cover Classification: Improving Minority Classes’
  Prediction Accuracy Using the Geometric SMOTE Algorithm. Remote Sensing,
  11(24), 3040. <https://doi.org/10.3390/rs11243040>
