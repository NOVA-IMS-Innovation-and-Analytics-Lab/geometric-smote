from sklearn import linear_model, svm, tree, neighbors
from sklearn.base import is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from ..metalearners import StackClassifier, StackRegressor, SuperLearnerRegressor, SuperLearnerClassifier
from ..validation import META_CLASSIFIER, META_REGRESSOR, CLASSIFIERS, REGRESSORS
from numpy.testing import assert_raises


# Simulate binary class, multi class and regression data
binary = make_classification(n_samples=1000, n_redundant=0)
X_bin, y_bin = binary[0], binary[1]

multiclass = make_classification(n_samples=1000, n_classes=3, n_informative = 10, n_redundant=0)
X_multi, y_multi = multiclass[0], multiclass[1]

regression = make_regression(n_samples=1000)
X_reg, y_reg = regression[0], regression[1]


def test_if_used_correct_metaestimator_type():
    """Tests if correct default metaestimator type
    is used for regression and classification output."""

    sc = StackClassifier()
    sc.fit(X_bin, y_bin)
    assert is_classifier(sc.meta_estimator_.model)

    sc = StackClassifier()
    sc.fit(X_multi, y_multi)
    assert is_classifier(sc.meta_estimator_.model)

    sr = StackRegressor()
    sr.fit(X_reg, y_reg)
    assert is_regressor(sr.meta_estimator_.model)

def test_incompatibility_data_and_estimators():
    """Tests if an error is raised when incompatible 
    estimators to the target data are used."""

    sr = StackRegressor(meta_estimator=META_CLASSIFIER)
    assert_raises(TypeError, sr.fit, X_reg, y_reg)

    sr = StackRegressor(estimators=CLASSIFIERS)
    assert_raises(TypeError, sr.fit, X_reg, y_reg)

    sr = StackRegressor()
    assert_raises(TypeError, sr.fit, X_bin, y_bin)

    sc = StackClassifier(meta_estimator=META_REGRESSOR)
    assert_raises(TypeError, sc.fit, X_bin, y_bin)

    sc = StackClassifier(estimators=REGRESSORS)
    assert_raises(TypeError, sc.fit, X_bin, y_bin)

    sc = StackClassifier()
    assert_raises(TypeError, sc.fit, X_reg, y_reg)

    slr = SuperLearnerRegressor(estimators=CLASSIFIERS)
    assert_raises(TypeError, slr.fit, X_reg, y_reg)

    slr = SuperLearnerRegressor()
    assert_raises(TypeError, slr.fit, X_bin, y_bin)

    slc = SuperLearnerClassifier(estimators=REGRESSORS)
    assert_raises(TypeError, slc.fit, X_bin, y_bin)

    slc = SuperLearnerClassifier()
    assert_raises(TypeError, slc.fit, X_reg, y_reg)

def test_not_list_of_tuples():
    """Checks if an error is raised when `meta_estimator` is not 
    a (string, estimator) tuple."""

    sr = StackRegressor(meta_estimator=META_REGRESSOR[0])
    assert_raises(TypeError, sr.fit, X_reg, y_reg)

    sc = StackClassifier(meta_estimator=META_CLASSIFIER[0])
    assert_raises(TypeError, sc.fit, X_bin, y_bin)

def test_predict_proba():
    """Checks if predict_proba returns the correct shape."""
 
    sc = StackClassifier()
    sc.fit(X_multi, y_multi)
    assert sc.predict_proba(X_multi).shape == (y_multi.size, 3)

    slc = SuperLearnerClassifier()
    slc.fit(X_multi, y_multi)
    assert slc.predict_proba(X_multi).shape == (y_multi.size, 3)

def test_predict():
    """Checks if predict returns the correct shape."""

    sr = StackRegressor()
    sr.fit(X_reg, y_reg)
    assert sr.predict(X_reg).shape == (y_reg.size, )

    slr = SuperLearnerRegressor()
    slr.fit(X_reg, y_reg)
    assert slr.predict(X_reg).shape == (y_reg.size, )