[scikit-learn]: <http://scikit-learn.org/stable/>
[imbalanced-learn]: <http://imbalanced-learn.org/stable/>
[BorderlineSMOTE]: <https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.BorderlineSMOTE.html>
[SMOTE]: <https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html>
[KMeans]:  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>
[DBSCAN]:  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>

# User guide

`imbalanced-learn-extra` is a Python package that extends [imbalanced-learn]. It implements algorithms that are not included in
[imbalanced-learn] due to their novelty or lower citation number. The current version includes the following:

- A general interface for clustering-based oversampling algorithms that introduces the `ClusterOverSampler` class, while
  `KMeansSMOTE`, `SOMO` and `GeometricSOMO` classes are provided for convinience. The distribution of the generated samples to the
  clusters is controled by the `distributor` parameter with `DensityDistributor` being an example of distribution that is based on
  the density of the clusters.

- The Geometric SMOTE algorithm as a geometrically enhanced drop-in replacement for SMOTE, that handles numerical as well as
categorical features.

## Clustering-based oversamping

Initially, we generate multi-class imbalanced data represented by the imput data `X` and targets `y`:

```python
>>> from collections import Counter
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_classes=3, weights=[0.10, 0.10, 0.80], random_state=0, n_informative=10)
>>> print(sorted(Counter(y).items()))
[(np.int64(0), 10), (np.int64(1), 10), (np.int64(2), 80)]
```

Below we provided some examples of the `imblearn_extra.clover` functionality.

### KMeans-SMOTE algorithm

KMeans-SMOTE[^2] algorithm is a combination of [KMeans] clusterer and [SMOTE] oversampler and it is implemented by the
`KMeansSMOTE` class. We initialize it with the default parameters and use it to resample the data:

```python
>>> from imblearn_extra.clover.over_sampling import KMeansSMOTE
>>> kmeans_smote = KMeansSMOTE(random_state=5)
>>> X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)
>>> print(sorted(Counter(y_resampled).items()))
[(np.int64(0), 80), (np.int64(1), 80), (np.int64(2), 80)]
```

The augmented data set can be used instead of the original data set to train a classifier:

```python
>>> from sklearn.tree import DecisionTreeClassifier
>>> clf = DecisionTreeClassifier()
>>> clf.fit(X_resampled, y_resampled)
DecisionTreeClassifier()
```

### Combining clusterers and oversamplers

The `ClusterOverSampler` class allows to combine [imbalanced-learn] oversamplers with [scikit-learn] clusterers. This achieved
through the use of the parameters `oversampler` and `clusterer`. For example, we can select [BorderlineSMOTE] as the oversampler
and [DBSCAN] as the clustering algorithm:

```python
>>> from sklearn.cluster import DBSCAN
>>> from imblearn.over_sampling import BorderlineSMOTE
>>> from imblearn_extra.clover.over_sampling import ClusterOverSampler
>>> dbscan_bsmote = ClusterOverSampler(oversampler=BorderlineSMOTE(random_state=5), clusterer=DBSCAN())
>>> X_resampled, y_resampled = dbscan_bsmote.fit_resample(X, y)
>>> print(sorted(Counter(y_resampled).items()))
[(np.int64(0), 80), (np.int64(1), 80), (np.int64(2), 80)]
```

Additionally, if the clusterer supports a neighboring structure for the clusters through a `neighbors_` attribute, then it can
be used to generate inter-cluster artificial data similarly to SOMO[^1] and G-SOMO[^3] algorithms.

### Adjusting the distribution of generated samples

The parameter `distributor` of the `ClusterOverSampler` is used to define the distribution of the generated samples to the
clusters. The `DensityDistributor` class implements a density based distribution and it is the default `distributor` for all
objects of the `ClusterOverSampler` class:

```python
>>> from sklearn.cluster import AgglomerativeClustering
>>> from imblearn.over_sampling import SMOTE
>>> agg_smote = ClusterOverSampler(oversampler=SMOTE(random_state=5), clusterer=AgglomerativeClustering())
>>> agg_smote.fit(X, y)
>>> agg_smote.distributor_
DensityDistributor()
```

The `DensityDistributor` objects can be parametrized:

```python
>>> from imblearn_extra.clover.distribution import DensityDistributor
>>> distributor = DensityDistributor(distances_exponent=0)
```

In order to distribute the samples a `labels` parameter is required, while `neighbors` is optional:

```python
>>> from sklearn.cluster import KMeans
>>> clusterer = KMeans(n_clusters=4, random_state=1).fit(X, y)
>>> labels = clusterer.labels_
```

The distribution samples of the samples is provided by the `fit_distribute` method and it is described in the `intra_distribution`
and `inter_distribution` dictionaries:

```python
>>> intra_distribution, inter_distribution = distributor.fit_distribute(X, y, labels, neighbors=None)
>>> print(distributor.filtered_clusters_)
[(np.int32(3), np.int64(1)), (np.int32(1), np.int64(0)), (np.int32(1), np.int64(1))]
>>> print(distributor.clusters_density_)
{(np.int32(3), np.int64(1)): np.float64(3.0), (np.int32(1), np.int64(0)): np.float64(7.0), (np.int32(1), np.int64(1)): np.float64(7.0)}
>>> print(intra_distribution)
{(np.int32(3), np.int64(1)): np.float64(0.7), (np.int32(1), np.int64(0)): np.float64(1.0), (np.int32(1), np.int64(1)): np.float64(0.3)}
>>> print(inter_distribution)
{}
```

The keys of the above dictionaries are tuples of `(cluster_label, class_label)` shape, while their values are proportions of the
total generated samples for the particular class. For example `(0, 1): 0.7` means that 70% of samples of class `1` will be
generated in the cluster `0`. Any other distributor can be defined by extending the `BaseDistributor` class.

## Geometric SMOTE algorithm

SMOTE algorithm, as well as any other over-sampling method based on the SMOTE mechanism, generates synthetic samples along line
segments that join minority class instances. Geometric SMOTE[^4] (G-SMOTE) is an enhancement of the SMOTE data generation
mechanism. G-SMOTE generates synthetic samples in a geometric region of the input space, around each selected minority instance.
The `GeometricSMOTE` class can be used with multiple classes as well as binary classes classification. It uses a one-vs-rest
approach by selecting each targeted class and computing the necessary statistics against the rest of the data set which are
grouped in a single class.

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
>>> from imblearn_extra.gsmote import GeometricSMOTE
>>> geometric_smote = GeometricSMOTE()
>>> X_resampled, y_resampled = geometric_smote.fit_resample(X, y)
>>> from collections import Counter
>>> print(sorted(Counter(y_resampled).items()))
[(np.int64(0), 80), (np.int64(1), 80), (np.int64(2), 80)]
```

The augmented data set can be used instead of the original data set to train a classifier:

```python
>>> from sklearn.tree import DecisionTreeClassifier
>>> clf = DecisionTreeClassifier()
>>> clf.fit(X_resampled, y_resampled)
DecisionTreeClassifier()
```

`GeometricSMOTE` can be used also in a machine learning pipeline:

```python
from imblearn.pipeline import make_pipeline
pipeline = make_pipeline(GeometricSMOTE(), DecisionTreeClassifier())
pipeline.fit(X, y)
Pipeline(steps=[('geometricsmote', GeometricSMOTE()),
                ('decisiontreeclassifier', DecisionTreeClassifier())])
```

### Compatibility

The API of `imblearn_extra` is fully compatible to [imbalanced-learn]. Particularly for clustering-based oversampling, any
oversampler from cluster-over-sampling that does not use clustering, i.e. when ``clusterer=None``, is equivalent to the
corresponding [imbalanced-learn] oversampler:

```python
>>> import numpy as np
>>> X_res_im, y_res_im = SMOTE(random_state=5).fit_resample(X, y)
>>> X_res_cl, y_res_cl = ClusterOverSampler(SMOTE(random_state=5), clusterer=None).fit_resample(X, y)
>>> np.testing.assert_equal(X_res_im, X_res_cl)
>>> np.testing.assert_equal(y_res_im, y_res_cl)
```

## References

[^1]: [G. Douzas, F. Bacao, "Self-Organizing Map Oversampling (SOMO) for imbalanced data set learning", Expert Systems with
    Applications, vol. 82, pp. 40-52, 2017.](https://www.sciencedirect.com/science/article/abs/pii/S0957417417302324)  
[^2]: [G. Douzas, F. Bacao, F. Last, "Improving imbalanced learning through a heuristic oversampling method based on k-means and SMOTE", Information Sciences, vol. 465, pp. 1-20,
    2018.](https://www.sciencedirect.com/science/article/abs/pii/S0020025518304997)  
[^3]: [G. Douzas, F. Bacao, F. Last, "G-SOMO: An oversampling approach based on self-organized maps and geometric SMOTE", Expert
    Systems with Applications, vol. 183,115230, 2021.](https://www.sciencedirect.com/science/article/abs/pii/S095741742100662X)  
[^4]: [G. Douzas, F. Bacao, F. Last, "Geometric SMOTE a geometrically enhanced drop-in replacement for SMOTE", Information
    Sciences, Volume 501, 2019.](https://www.sciencedirect.com/science/article/abs/pii/S0020025519305353?via%3Dihub)  
