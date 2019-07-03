"""
===============================================================
Customized sampler to implement an outlier rejections estimator
===============================================================

This example illustrates the use of a custom sampler to implement an outlier
rejections estimator. It can be used easily within a pipeline in which the
number of samples can vary during training, which usually is a limitation of
the current scikit-learn pipeline.

"""

from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced
import sklearn.datasets as datasets
from imblearn.pipeline import make_pipeline
from gsmote import GeometricSMOTE
from sklearn.ensemble import RandomForestClassifier

# Fetch data
X, y, description = datasets.fetch_covtype().values()
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1)
print(description)

# Display an overview of observations per class
counts = dict(Counter(y))
pd.DataFrame(counts.values(), index=counts.keys(), columns=['count']).sort_index()

# Set up experiments using G-SMOTE and no over sampling
clf = RandomForestClassifier(bootstrap=True)
ovs_clf = make_pipeline(GeometricSMOTE(), RandomForestClassifier(bootstrap=True))

for comb in [clf, ovs_clf]:
    title = f'{comb.__class__.__name__} - Results'
    div = '='*len(title)
    print(div+'\n'+title+'\n'+div)
    comb.fit(X_train, y_train)
    y_pred_bal = comb.predict(X_test)
    print(classification_report_imbalanced(y_test, y_pred_bal))
