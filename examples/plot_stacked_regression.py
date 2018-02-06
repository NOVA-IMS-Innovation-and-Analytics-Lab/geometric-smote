"""
========================
Super Learner regression
========================

Simulated 1D regression data with stacked regressions.

Four different simulations from reference [3] are presented. For 
each simulation a stacked regressor is fitted to the data using 
the default estimators (base estimators and meta-estimator) and the 
true model, the simulated data and the superlearner fit are ploted.
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
from stacklearn.metalearners import StackRegressor

# The simulated data
X = np.random.uniform(low=-4, high=4, size=100)
X_sorted = np.sort(X).reshape(-1, 1)
X_plot = np.arange(-4, 4, 0.01)
e = np.random.normal(size = 100)

def curve1(X, e):
    return -2 * np.int64(X < -3) + 2.55 * np.int64(X > -2) - 2 * np.int64(X > 0) + 4 * np.int64(X > 2) - np.int64(X > 3) + e

def curve2(X, e):
    return 6 + 0.4 * X - 0.36 * X ** 2 + 0.005 * X ** 3 + e

def curve3(X, e):
    return 2.83 * np.sin(np.pi * X / 2) + e

def curve4(X, e): 
    return 4 * np.sin(3 * np.pi * X) * np.int64(X > 0) + e

curves = [curve1, curve2, curve3, curve4]

# Create superlearner regressor
sr = StackRegressor()

# Plots
for curve_id, curve in enumerate(curves):
    plt.subplot(2, 2, curve_id + 1)
    sr.fit(X.reshape(-1, 1), curve(X, e))
    plt.plot(X, curve(X, e), "bo", X_plot, curve(X_plot, 0), "r", X_sorted, sr.predict(X_sorted), "g--", ms=2)
    plt.xlabel("X")
    plt.ylabel("Y")
plt.legend(("Simulated points", "True model", "Stacked regressors fit"), ncol=3, bbox_to_anchor=(0, -0.1), loc="upper center")
plt.show()