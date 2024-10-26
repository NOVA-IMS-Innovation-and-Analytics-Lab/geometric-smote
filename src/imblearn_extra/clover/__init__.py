"""A general interface for clustering based over-sampling algorithms.

[SOMO oversampling algorithm]: <https://www.sciencedirect.com/science/article/abs/pii/S0957417417302324>
[KMeans-SMOTE oversampling algorithm]: <https://www.sciencedirect.com/science/article/abs/pii/S0020025518304997>
[G-SOMO oversampling algorithm]: <https://www.sciencedirect.com/science/article/abs/pii/S095741742100662X>

The module provides the implementation of an interface for clustering-based over-sampling. It
has two submodules:

- [`distribution`][imblearn_extra.clover.distribution]: Provides the classes to distrubute the generated samples into
clusters.

    - [`DensityDistributor`][imblearn_extra.clover.distribution.DensityDistributor]: Density based distributor.

- [`over_sampling`][imblearn_extra.clover.over_sampling]: Provides the clustering-based oversampling algorithms.

    - [`ClusterOverSampler`][imblearn_extra.clover.over_sampling.ClusterOverSampler]: Combinations of oversampler and
    clusterer.
    - [`KMeansSMOTE`][imblearn_extra.clover.over_sampling.KMeansSMOTE]: [KMeans-SMOTE oversampling algorithm]
    oversampling algorithm.
    - [`SOMO`][imblearn_extra.clover.over_sampling.SOMO]: [SOMO oversampling algorithm].
    - [`GeometricSOMO`][imblearn_extra.clover.over_sampling.GeometricSOMO]: [G-SOMO oversampling algorithm].
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__: list[str] = []

InputData = npt.NDArray[np.float64]
Targets = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int16]
Neighbors = npt.NDArray[np.int16]
MultiLabel = tuple[int, int]
IntraDistribution = dict[MultiLabel, float]
InterDistribution = dict[tuple[MultiLabel, MultiLabel], float]
Density = dict[MultiLabel, float]
