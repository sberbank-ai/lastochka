# -*- coding: utf-8 -*-
# this is hack to import from pre-parent directory
import unittest
import pandas as pd
from lastochka import LastochkaTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lastochka.core.functions import check_mono


class BaseTest(unittest.TestCase):
    """
    Basic functionality tests
    """

    def test_basic(self):
        N_SAMPLES = 1000

        X, y = make_classification(n_samples=N_SAMPLES, n_features=10, n_informative=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        column_names = ['X%i' % i for i in range(10)]

        D_train = pd.DataFrame(X_train, columns=column_names)
        D_test = pd.DataFrame(X_test, columns=column_names)

        lastochka = LastochkaTransformer()
        log = LogisticRegression()

        pipe = Pipeline(steps=[
                ('lastochka', lastochka),
                ('log', log)])

        pipe.fit(D_train, y_train)

        for variable, transformer in lastochka.fitted_wing.items():
            is_mono = check_mono(transformer.cont_df_woe['woe'].values)
            self.assertTrue(is_mono, msg="Variable %s is NOT monothonized" % variable)

        pipe.predict_proba(D_test)


if __name__ == "__main__":
    unittest.main()
