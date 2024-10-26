[scikit-learn]: <http://scikit-learn.org/stable/>
[imbalanced-learn]: <http://imbalanced-learn.org/stable/>
[SOMO]: <https://www.sciencedirect.com/science/article/abs/pii/S0957417417302324>
[KMeans-SMOTE]: <https://www.sciencedirect.com/science/article/abs/pii/S0020025518304997>
[G-SOMO]: <https://www.sciencedirect.com/science/article/abs/pii/S095741742100662X>
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
[version badge]: <https://img.shields.io/pypi/v/imbalanced-learn-extra.svg>
[pythonversion badge]: <https://img.shields.io/pypi/pyversions/imbalanced-learn-extra.svg>
[downloads badge]: <https://img.shields.io/pypi/dd/imbalanced-learn-extra>
[gitter]: <https://gitter.im/imbalanced-learn-extra/community>
[gitter badge]: <https://badges.gitter.im/join%20chat.svg>
[discussions]: <https://github.com/georgedouzas/imbalanced-learn-extra/discussions>
[discussions badge]: <https://img.shields.io/github/discussions/georgedouzas/imbalanced-learn-extra>
[ci]: <https://github.com/georgedouzas/imbalanced-learn-extra/actions?query=workflow>
[ci badge]: <https://github.com/georgedouzas/imbalanced-learn-extra/actions/workflows/ci.yml/badge.svg?branch=main>
[doc]: <https://github.com/georgedouzas/imbalanced-learn-extra/actions?query=workflow>
[doc badge]: <https://github.com/georgedouzas/imbalanced-learn-extra/actions/workflows/doc.yml/badge.svg?branch=main>

# imbalanced-learn-extra

[![ci][ci badge]][ci] [![doc][doc badge]][doc]

| Category          | Tools    |
| ------------------| -------- |
| **Development**   | [![black][black badge]][black] [![ruff][ruff badge]][ruff] [![mypy][mypy badge]][mypy] [![docformatter][docformatter badge]][docformatter] |
| **Package**       | ![version][version badge] ![pythonversion][pythonversion badge] ![downloads][downloads badge] |
| **Documentation** | [![mkdocs][mkdocs badge]][mkdocs]|
| **Communication** | [![gitter][gitter badge]][gitter] [![discussions][discussions badge]][discussions] |

## Introduction

`imbalanced-learn-extra` is a Python package that extends [imbalanced-learn]. It implements algorithms that are not included in
[imbalanced-learn] due to their novelty or lower citation number. The current version includes the following:

- A general interface for clustering-based oversampling algorithms.

- The Geometric SMOTE algorithm. It is a geometrically enhanced drop-in replacement for SMOTE, that handles numerical as well as
categorical features.

## Installation

For user installation, `imbalanced-learn-extra` is currently available on the PyPi's repository, and you can
install it via `pip`:

```bash
pip install imbalanced-learn-extra
```

Development installation requires cloning the repository and then using [PDM](https://github.com/pdm-project/pdm) to install the
project as well as the main and development dependencies:

```bash
git clone https://github.com/georgedouzas/imbalanced-learn-extra.git
cd imbalanced-learn-extra
pdm install
```

SOM clusterer requires optional dependencies:

```bash
pip install imbalanced-learn-extra[som]
```

## Usage

All the classes included in `imbalanced-learn-extra` follow the [imbalanced-learn] API using the functionality of the base
oversampler. Using [scikit-learn] convention, the data are represented as follows:

- Input data `X`: 2D array-like or sparse matrices.
- Targets `y`: 1D array-like.

The oversamplers implement a `fit` method to learn from `X` and `y`:

```python
oversampler.fit(X, y)
```

They also implement a `fit_resample` method to resample `X` and `y`:

```python
X_resampled, y_resampled = clustering_based_oversampler.fit_resample(X, y)
```

## Citing `imbalanced-learn-extra`

Publications using clustering-based oversampling:

- [G. Douzas, F. Bacao, "Self-Organizing Map Oversampling (SOMO) for imbalanced data set learning", Expert Systems with
    Applications, vol. 82, pp. 40-52, 2017.][SOMO]
- [G. Douzas, F. Bacao, F. Last, "Improving imbalanced learning through a heuristic oversampling method based on k-means and
    SMOTE", Information Sciences, vol. 465, pp. 1-20, 2018.][KMeans-SMOTE]
- [G. Douzas, F. Bacao, F. Last, "G-SOMO: An oversampling approach based on self-organized maps and geometric SMOTE", Expert
    Systems with Applications, vol. 183,115230, 2021.][G-SOMO]

Publications using Geometric-SMOTE:

- Douzas, G., Bacao, B. (2019). Geometric SMOTE: a geometrically enhanced
  drop-in replacement for SMOTE. Information Sciences, 501, 118-135.
  <https://doi.org/10.1016/j.ins.2019.06.007>

- Fonseca, J., Douzas, G., Bacao, F. (2021). Increasing the Effectiveness of
  Active Learning: Introducing Artificial Data Generation in Active Learning
  for Land Use/Land Cover Classification. Remote Sensing, 13(13), 2619.
  <https://doi.org/10.3390/rs13132619>

- Douzas, G., Bacao, F., Fonseca, J., Khudinyan, M. (2019). Imbalanced
  Learning in Land Cover Classification: Improving Minority Classes’
  Prediction Accuracy Using the Geometric SMOTE Algorithm. Remote Sensing,
  11(24), 3040. <https://doi.org/10.3390/rs11243040>
