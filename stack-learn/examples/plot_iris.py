"""
==============================================================
Plot Super Learner and stacked classifiers in the iris dataset
==============================================================

Comparison of Super Learner and Stacked classifiers on a 
2D projection of the iris dataset. We only consider the 
first 2 features of this dataset:

- Sepal length
- Sepal width

The two metaestimators use the same set of base estimators.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from stacklearn.metalearners import SuperLearnerClassifier, StackClassifier
from stacklearn.validation import CLASSIFIERS

# Iris data
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Create a mesh
plot_step = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

# Create metalearning classifiers
slc = SuperLearnerClassifier().fit(X, y)
sc = StackClassifier().fit(X, y)

# Title for the plots
titles = ["Super Learner classifier", "Stacked Classifiers"]

# Plot the decision boundaries
for i, clf in enumerate((slc, sc)):
    
    plt.subplot(1, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()