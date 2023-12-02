[scikit-learn]: <http://scikit-learn.org/stable/>
[imbalanced-learn]: <http://imbalanced-learn.org/stable/>
[BorderlineSMOTE]: <https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html>
[SMOTE]: <https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html>
[KMeans]:  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
[DBSCAN]:  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>

# User guide

SMOTE algorithm, as well as any other over-sampling method based on the SMOTE mechanism, generates synthetic samples along line
segments that join minority class instances. Geometric SMOTE (G-SMOTE) is an enhancement of the SMOTE data generation mechanism.
G-SMOTE generates synthetic samples in a geometric region of the input space, around each selected minority instance. The
`GeometricSMOTE` class can be used with multiple classes as well as binary classes classification. It uses a one-vs-rest approach
by selecting each targeted class and computing the necessary statistics against the rest of the data set which are grouped in a
single class.

Initially, we generate multi-class imbalanced data represented by the input data `X` and targets `y`:

```python
>>> from collections import Counter
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_classes=3, weights=[0.10, 0.10, 0.80], random_state=0, n_informative=10)
>>> print(sorted(Counter(y).items()))
[(0, 10), (1, 10), (2, 80)]
```

We can use `GeometricSMOTE` to resample the data:

```python
>>> from gsmote import GeometricSMOTE
>>> geometric_smote = GeometricSMOTE()
>>> X_resampled, y_resampled = geometric_smote.fit_resample(X, y)
>>> from collections import Counter
>>> print(sorted(Counter(y_resampled).items()))
[(0, 80), (1, 80), (2, 80)]
```

The augmented data set can be used instead of the original data set to train a classifier:

```python
>>> from sklearn.tree import DecisionTreeClassifier
>>> clf = DecisionTreeClassifier()
>>> clf.fit(X_resampled, y_resampled)
```

`GeometricSMOTE` can be used also in a machine learning pipeline:

```python
from imblearn.pipeline import make_pipeline
pipeline = make_pipeline(GeometricSMOTE(), DecisionTreeClassifier())
pipeline.fit(X, y)
```
