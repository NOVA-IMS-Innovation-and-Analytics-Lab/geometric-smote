"""
Implement standard oversamplers and monkey patch their methods.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from imblearn import over_sampling

from .base import ExtendedBaseOverSampler

SMOTE_ATTRIBUTES = (
    '_in_danger_noise', 
    '_make_samples', 
    '_validate_estimator', 
    '_sample_regular', 
    '_sample_borderline', 
    '_sample_svm'
)


class SMOTE(ExtendedBaseOverSampler):

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 categorical_cols=None,
                 clusterer=None,
                 distribution_function=None,
                 k=None,
                 k_neighbors=5,
                 m=None,
                 m_neighbors=10,
                 out_step=0.5,
                 kind='regular',
                 svm_estimator=None,
                 n_jobs=1):
        super().__init__(ratio=ratio,
                         random_state=random_state,
                         categorical_cols=categorical_cols,
                         clusterer=clusterer,
                         distribution_function=distribution_function)
        self.kind = kind
        self.k = k
        self.k_neighbors = k_neighbors
        self.m = m
        self.m_neighbors = m_neighbors
        self.out_step = out_step
        self.svm_estimator = svm_estimator
        self.n_jobs = n_jobs
    
    def _basic_sample(self, X, y):
        pass

for attribute in SMOTE_ATTRIBUTES:
    setattr(SMOTE, attribute, getattr(over_sampling.SMOTE, attribute))
SMOTE._basic_sample = over_sampling.SMOTE._sample