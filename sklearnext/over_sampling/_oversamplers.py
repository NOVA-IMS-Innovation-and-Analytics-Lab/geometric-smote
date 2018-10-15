"""
Implement standard oversamplers and monkey patch their methods.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from imblearn import over_sampling

from .base import ExtendedBaseOverSampler


def monkey_patch_attributes(attributes_mapping):
    """Monkey patch attributes to the selected oversamplers."""
    for (patched_oversampler, oversampler), attributes in attributes_mapping.items():
        for attribute in attributes_mapping[(patched_oversampler, oversampler)]:
            setattr(patched_oversampler, attribute, getattr(oversampler, attribute))
        patched_oversampler._basic_sample = oversampler._sample


class RandomOverSampler(ExtendedBaseOverSampler):

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 categorical_cols=None,
                 clusterer=None,
                 distributor=None):
        super().__init__(ratio=ratio,
                         random_state=random_state,
                         categorical_cols=categorical_cols,
                         clusterer=clusterer,
                         distributor=distributor)
    
    def _basic_sample(self, X, y):
        pass


class SMOTE(ExtendedBaseOverSampler):

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 categorical_cols=None,
                 clusterer=None,
                 distributor=None,
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
                         distributor=distributor)
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


class ADASYN(ExtendedBaseOverSampler):

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 categorical_cols=None,
                 clusterer=None,
                 distributor=None,
                 k=None,
                 n_neighbors=5,
                 n_jobs=1):
        super().__init__(ratio=ratio,
                         random_state=random_state,
                         categorical_cols=categorical_cols,
                         clusterer=clusterer,
                         distributor=distributor)
        self.k = k
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
    
    def _basic_sample(self, X, y):
        pass


# Define monkey patched attributes
ATTRIBUTES_MAPPING = {
    (RandomOverSampler, over_sampling.RandomOverSampler): (),
    (SMOTE, over_sampling.SMOTE): ('_in_danger_noise', '_make_samples', '_validate_estimator', '_sample_regular', '_sample_borderline', '_sample_svm'),
    (ADASYN, over_sampling.ADASYN): ('_validate_estimator', )
}

# Apply monkey patching
monkey_patch_attributes(ATTRIBUTES_MAPPING)
