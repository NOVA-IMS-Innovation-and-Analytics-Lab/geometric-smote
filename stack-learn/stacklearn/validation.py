"""
This module contains various functions used in metafeatures and metalerning modules.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>

from sklearn import linear_model, svm, tree, neighbors, kernel_ridge, discriminant_analysis, ensemble
from sklearn.base import TransformerMixin, RegressorMixin, ClassifierMixin, clone, is_classifier
from sklearn.utils.multiclass import type_of_target, unique_labels
from collections import namedtuple


estimator_tuple = namedtuple("estimator_tuple", ["name", "model"])

CLASSIFIERS = [("lr", linear_model.LogisticRegression()),
               ("lda", discriminant_analysis.LinearDiscriminantAnalysis()),
               ("svc", svm.SVC(probability=True)), 
               ("gbc", ensemble.GradientBoostingClassifier())]

REGRESSORS = [("rr", linear_model.Ridge()), 
              ("lars", linear_model.Lars()), 
              ("svr", svm.SVR()), 
              ("gbr", ensemble.GradientBoostingRegressor())]

META_CLASSIFIER = ("lr", linear_model.LogisticRegression())

META_REGRESSOR = ("rr", linear_model.Ridge())

def _check_estimator_format(estimator):
    if not isinstance(estimator, tuple):
        raise TypeError("Estimator should be a (string, estimator) tuple.")
    if len(estimator) != 2:
        raise TypeError("The estimator tuple should have two components (string, estimator).")
    if not isinstance(estimator[0], str):
        raise TypeError("The first component of the tuple should be of type str.")
    if not isinstance(estimator[1], (ClassifierMixin, RegressorMixin)):
        raise TypeError("The second component of the tuple should be an classifier or a regressor.")
    return estimator_tuple(estimator[0], clone(estimator[1]))

def _check_estimator_target(estimator, type_of_target, force_probability = True):
    if is_classifier(estimator.model):
        if type_of_target is "continuous":
            raise TypeError("Classifier %s is not allowed for a continuous target." % estimator.name)
        if force_probability and "probability" in estimator.model.get_params():
            estimator.model.probability = True
    else:
        if type_of_target is not "continuous":
            raise TypeError("Regressor %s is not allowed for a binary or multiclass target." % estimator.name)
        return estimator

def _check_unique_names(estimators):
    names = [estimator.name for estimator in estimators]
    if len(set(names)) < len(names):
        raise ValueError("Names of base estimators should be unique.")
    return estimators

def _check_sample_weight(estimator, sample_weight):
    if sample_weight is not None and not has_fit_parameter(estimator.model, "sample_weight"):
        raise ValueError("Metaestimator %s does not support sample weights." % estimator.name)
    return estimator

def check_type_of_target(y):
    """Checks if type of target is binary, multiclass or continuous 
    and returns the type of target and the number of class labels."""

    tot = type_of_target(y)
    if tot not in ("binary", "multiclass", "continuous"):
        raise TypeError("Only binary, multiclass or continuous output is supported.")
    
    n_classes = None
    if tot is not "continuous":
        n_classes = len(unique_labels(y))

    return tot, n_classes

def check_base_estimators(estimators, type_of_target):
    """Checks the base estimators and compares their type 
    to the type of target."""
    
    # If estimators is a list, clone and check each element
    if isinstance(estimators, list):
        estimators_ = []
        for estimator in estimators:
            estimator_ = _check_estimator_format(estimator)
            estimator_ = _check_estimator_target(estimator_, type_of_target)
            estimators_.append(estimator_)
        estimators_ = _check_unique_names(estimators_)
    # Set default estimators
    elif estimators is None:
        if type_of_target is "continuous":
            estimators_ = [_check_estimator_format(estimator) for estimator in REGRESSORS]
        else:
            estimators_ = [_check_estimator_format(estimator) for estimator in CLASSIFIERS]
    else:
        raise TypeError("Parameter `estimators` should be a list of (string, estimator) tuples.")
    
    return estimators_

def check_all_estimators(metaestimator, estimators, is_classification_task, sample_weight):
    """Checks the type of the metaestimator."""

    # Check base estimators
    for estimator in estimators:
        if is_classification_task and not is_classifier(estimator.model):
            raise TypeError("Base estimator %s should be a classifier." % estimator.name)
        if not is_classification_task and is_classifier(estimator.model):
            raise TypeError("Base estimator %s should be a regressor." % estimator.name)
    
    # Clone and check metaestimator
    if metaestimator is not None:
        metaestimator_ = _check_estimator_format(metaestimator)
        if is_classification_task and not is_classifier(metaestimator_.model):
            raise TypeError("Metaestimator %s should be a classifier." % metaestimator_.name)
        if not is_classification_task and is_classifier(metaestimator_.model):
            raise TypeError("Metaestimator %s should be a regressor." % metaestimator_.name)
        
    # Set default estimators    
    else:
        if is_classification_task:
            metaestimator_ = _check_estimator_format(META_CLASSIFIER)
        else:
            metaestimator_ = _check_estimator_format(META_REGRESSOR)
    metaestimator_ = _check_sample_weight(metaestimator_, sample_weight)
    return metaestimator_