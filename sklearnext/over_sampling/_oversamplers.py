"""
The :mod:`sklearnext.over_sampling._oversamplers` reimplements 
standard oversamplers and monkey patch their methods to make them 
compatible with the clustering based oversampling API.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: BSD 3 clause

from imblearn.over_sampling import (
    RandomOverSampler as _RandomOverSampler,
    SMOTE as _SMOTE,
    BorderlineSMOTE as _BorderlineSMOTE,
    SVMSMOTE as _SVMSMOTE,
    SMOTENC as _SMOTENC,
    ADASYN as _ADASYN
)
from imblearn.utils import check_target_type, check_neighbors_object
from sklearn.utils import check_X_y

from .base import BaseClusterOverSampler


def monkey_patch_attributes(attributes_mapping):
    """Monkey patch attributes to the selected oversamplers."""
    for (patched_oversampler, oversampler), attributes in attributes_mapping.items():
        for attribute in attributes_mapping[(patched_oversampler, oversampler)]:
            setattr(patched_oversampler, attribute, getattr(oversampler, attribute))


class RandomOverSampler(BaseClusterOverSampler):

    def __init__(self,
                 sampling_strategy='auto',
                 clusterer=None,
                 distributor=None,
                 return_indices=False,
                 random_state=None,
                 ratio=None):
        super(RandomOverSampler, self).__init__(sampling_strategy=sampling_strategy, clusterer=clusterer, 
                                                distributor=distributor, ratio=ratio)
        self.return_indices = return_indices
        self.random_state = random_state
    
    def _basic_sample(self, X, y):
        return _RandomOverSampler._fit_resample(self, X, y)


class SMOTE(BaseClusterOverSampler):

    def __init__(self,
                 sampling_strategy='auto',
                 clusterer=None,
                 distributor=None,
                 random_state=None,
                 k_neighbors=5,
                 m_neighbors='deprecated',
                 out_step='deprecated',
                 kind='deprecated',
                 svm_estimator='deprecated',
                 n_jobs=1,
                 ratio=None):
        super(SMOTE, self).__init__(sampling_strategy=sampling_strategy, clusterer=clusterer,
                                    distributor=distributor, ratio=ratio)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.kind = kind
        self.m_neighbors = m_neighbors
        self.out_step = out_step
        self.svm_estimator = svm_estimator
        self.n_jobs = n_jobs
    
    def _basic_sample(self, X, y):
        return _SMOTE._fit_resample(self, X, y)


class BorderlineSMOTE(BaseClusterOverSampler, _BorderlineSMOTE):

    def __init__(self,
                 sampling_strategy='auto',
                 clusterer=None,
                 distributor=None,
                 random_state=None,
                 k_neighbors=5,
                 m_neighbors=10,
                 kind='borderline-1',
                 n_jobs=1):
        super(BorderlineSMOTE, self).__init__(sampling_strategy=sampling_strategy, clusterer=clusterer,
                                              distributor=distributor, ratio=None)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.kind=kind
        self.n_jobs = n_jobs
    
    def _basic_sample(self, X, y):
        return _BorderlineSMOTE._fit_resample(self, X, y)


class SVMSMOTE(BaseClusterOverSampler, _SVMSMOTE):

    def __init__(self,
                 sampling_strategy='auto',
                 clusterer=None,
                 distributor=None,
                 random_state=None,
                 k_neighbors=5,
                 m_neighbors=10,
                 svm_estimator=None,
                 out_step=0.5,
                 n_jobs=1):
        super(SVMSMOTE, self).__init__(sampling_strategy=sampling_strategy, clusterer=clusterer,
                                       distributor=distributor, ratio=None)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.svm_estimator = svm_estimator 
        self.out_step = out_step
        self.n_jobs = n_jobs
    
    def _basic_sample(self, X, y):
        return _SVMSMOTE._fit_resample(self, X, y)


class SMOTENC(_SMOTENC, BaseClusterOverSampler):

    def __init__(self,
                 categorical_features,
                 sampling_strategy='auto',
                 clusterer=None,
                 distributor=None,
                 random_state=None,
                 k_neighbors=5,
                 n_jobs=1):
        BaseClusterOverSampler.__init__(self, sampling_strategy=sampling_strategy, 
                                        clusterer=clusterer, distributor=distributor, ratio=None)
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs
        self.kind='deprecated'

    @staticmethod
    def _check_X_y(X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], dtype=None)
        return X, y, binarize_y
    
    def _basic_sample(self, X, y):
        return _SMOTENC._fit_resample(self, X, y)


class ADASYN(BaseClusterOverSampler, _ADASYN):

    def __init__(self,
                 sampling_strategy='auto',
                 clusterer=None,
                 distributor=None,
                 random_state=None,
                 n_neighbors=5,
                 n_jobs=1,
                 ratio=None):
        super(ADASYN, self).__init__(sampling_strategy=sampling_strategy, clusterer=clusterer, 
                                     distributor=distributor, ratio=ratio)
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
    
    def _basic_sample(self, X, y):
        return _ADASYN._fit_resample(self, X, y)


# Define monkey patched attributes
SMOTE_ATTRIBUTES = ('_validate_estimator', '_make_samples', '_generate_sample', '_in_danger_noise', '_sample')
ATTRIBUTES_MAPPING = {
    (RandomOverSampler, _RandomOverSampler): (),
    (SMOTE, _SMOTE): SMOTE_ATTRIBUTES,
    (BorderlineSMOTE, _BorderlineSMOTE): SMOTE_ATTRIBUTES,
    (SVMSMOTE, _SVMSMOTE): SMOTE_ATTRIBUTES,
    (SMOTENC, _SMOTENC): SMOTE_ATTRIBUTES,
    (ADASYN, _ADASYN): ('_validate_estimator', )
}

# Apply monkey patching
monkey_patch_attributes(ATTRIBUTES_MAPPING)
