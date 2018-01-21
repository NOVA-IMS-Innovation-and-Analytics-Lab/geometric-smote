from sklearn import svm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import make_classification, make_regression
from ..metafeatures import CrossValidationExtractor
from ..validation import CLASSIFIERS, REGRESSORS
from numpy.testing import assert_raises


# Simulate binary class, multi class and regression data
binary = make_classification(n_samples=1000, n_redundant=0)
X_bin, y_bin = binary[0], binary[1]

multiclass = make_classification(n_samples=1000, n_classes=3, n_informative = 10, n_redundant=0)
X_multi, y_multi = multiclass[0], multiclass[1]

regression = make_regression(n_samples=1000)
X_reg, y_reg = regression[0], regression[1]


def test_incompatibility_data_and_estimators():
    """Tests if an error is raised when classifiers are included in
    the list of estimators for a continuous target."""

    estimators = CLASSIFIERS + REGRESSORS
    cve = CrossValidationExtractor(estimators)
    assert_raises(TypeError, cve.fit_transform, X_reg, y_reg)

def test_not_list_of_tuples():
    """Checks if an error is raised when `estimators` is not a list 
    of (string, estimator) tuples."""

    estimators = CLASSIFIERS[0]
    cve = CrossValidationExtractor(estimators)
    assert_raises(TypeError, cve.fit_transform, X_bin, y_bin)

def test_type_of_cv_classification():
    """Checks if stratisfied k-fold cross validation 
    is used for binary or multiclass classification."""

    cve = CrossValidationExtractor()
    cve.fit_transform(X_bin, y_bin)
    assert isinstance(cve.cv_, StratifiedKFold)

    cve = CrossValidationExtractor()
    cve.fit_transform(X_multi, y_multi)
    assert isinstance(cve.cv_, StratifiedKFold)

def test_type_of_cv_regression():
    """Checks if k-fold cross validation is used for
    regression."""

    cve = CrossValidationExtractor()
    cve.fit_transform(X_reg, y_reg)
    assert isinstance(cve.cv_, KFold)

def test_predict_proba():
    """Checks if predict_proba is used."""
 
    cve = CrossValidationExtractor()
    assert cve.fit_transform(X_bin, y_bin).shape == (y_bin.size, 2 * len(cve.estimators_))

def test_predict():
    """Checks if predict is used."""

    cve = CrossValidationExtractor()
    assert cve.fit_transform(X_reg, y_reg).shape == (y_bin.size, len(cve.estimators_))