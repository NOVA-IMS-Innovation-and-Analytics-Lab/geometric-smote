import unittest
from sklearn.datasets import make_classification
from sklearnext.preprocessing.over_sampling.somo import SOMO
import random as r
import numpy as np


class TestingSOMO_MulticlassClassifier(unittest.TestCase):

    # Test: Random state initialization

    def test_randomState(self):
        rand_state = r.randint(0,1000)
        somo = SOMO(random_state=rand_state)
        self.assertEqual(somo.random_state,rand_state)

    # Test: correct output shape of generated data

    def test_output_X(self):
        len_X = r.randint(3,10)
        X, y = make_classification(n_classes=3, class_sep=2,
           weights=[0.7,0.3], n_informative=len_X, n_redundant=0, flip_y=0.1,
           n_features=len_X, n_clusters_per_class=1, n_samples=1000, random_state=10)
        somo = SOMO(som_rows=20,som_cols=20)
        X_res, y_res = somo.fit_sample(X, y)
        self.assertEqual((np.shape(X_res)[1]),len_X)


    # Test: Correct ratio of generated synthetic data

    def test_output_ratio(self):
        X, y = make_classification(n_classes=3, class_sep=2,
            weights=(0.6,0.2,0.2), n_informative=4, n_redundant=0, flip_y=0.1,
            n_features=5, n_clusters_per_class=1, n_samples=1000, random_state=None)
        somo = SOMO(ratio = 'auto', som_rows=20,som_cols=20)
        X_res, y_res = somo.fit_sample(X, y)
        count = [len(y_res[y_res==num]) for num in set(y_res)]
        ratio = [(num/max(count)) for num in count]
        self.assertTrue(0.99 < np.average(ratio) <= 1)


    # Test: Handle negative values

    def test_output_ratio(self):
        X, y = make_classification(n_classes=3, class_sep=2,
            weights=(0.6,0.2,0.2), n_informative=4, n_redundant=0, flip_y=0.1,
            n_features=5, n_clusters_per_class=1, n_samples=1000, random_state=None, shift = -10)
        self.assertTrue(X.mean() < 0)
        somo = SOMO(ratio = 'auto', som_rows=20,som_cols=20)
        X_res, y_res = somo.fit_sample(X, y)
        self.assertTrue(X_res.mean() < 0)


    #Test: Validate multiple target classes

    def test_multiple_classes(self):
        for num in range(2,10):
            X, y = make_classification(n_classes=num, class_sep=2, n_informative=4, n_redundant=0, flip_y=0.1,
                n_features=5, n_clusters_per_class=1, n_samples=1000, random_state=None)
            somo = SOMO(ratio = 'auto', som_rows=20,som_cols=20)
            X_res, y_res = somo.fit_sample(X, y)
            self.assertTrue(set(y) == set(y_res))



if __name__ == '__main__':
    unittest.main()
