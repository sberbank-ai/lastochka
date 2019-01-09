# -*- coding: utf-8 -*-
# this is hack to import from pre-parent directory
import unittest
import pandas as pd
import numpy as np
from lastochka import LastochkaTransformer
from lastochka.core.main import VectorTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class BaseTest(unittest.TestCase):
    """
    Basic functionality tests
    """

    def testBasic(self):
        N_SAMPLES = 200

        X, y = make_classification(n_samples=N_SAMPLES, n_features=10, n_informative=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        column_names = ['X%i' % i for i in range(10)]

        D_train = pd.DataFrame(X_train, columns=column_names)
        D_test = pd.DataFrame(X_test, columns=column_names)

        lastochka = LastochkaTransformer(verbose=True)
        log = LogisticRegression()

        pipe = Pipeline(steps=[
                ('lastochka', lastochka),
                ('log', log)])

        pipe.fit(D_train, y_train)

    def testNonExistentEstimator(self):
        vt = VectorTransformer(optimizer="fake_optimizer",
                               n_final=5, n_initial=10,
                               specials={}, verbose=False, name="test")

        self.assertRaises(NotImplementedError, vt.fit,
                          X=np.random.normal(size=100),
                          y=np.random.randint(0, 2, size=100))


if __name__ == "__main__":
    unittest.main()
