"""This module includes classes for clustering-based oversampling.

A general class for clustering-based oversampling as well as specific
clustering-based oversamplers are provided.
"""

from ._cluster import (
    ClusterOverSampler,
    clone_modify,
    extract_inter_data,
    extract_intra_data,
    generate_in_cluster,
    modify_nn,
)
from ._gsomo import GeometricSOMO
from ._kmeans_smote import KMeansSMOTE
from ._somo import SOMO

__all__: list[str] = [
    'ClusterOverSampler',
    'KMeansSMOTE',
    'SOMO',
    'GeometricSOMO',
    'modify_nn',
    'clone_modify',
    'extract_inter_data',
    'extract_intra_data',
    'generate_in_cluster',
]
