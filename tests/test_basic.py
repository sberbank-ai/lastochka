# -*- coding: utf-8 -*-
# this is hack to import from pre-parent directory
import unittest
import pandas as pd
import numpy as np
from lastochka.core.functions import calculate_stats
from lastochka import LastochkaTransformer
from lastochka.core.main import VectorTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score
from lastochka.core.optimizer import CategoryOptimizer


class BaseTest(unittest.TestCase):
    """
    Basic functionality tests
    """

    def testNumeric(self):
        N_SAMPLES = 200

        X, y = make_classification(n_samples=N_SAMPLES, n_features=10, n_informative=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
        column_names = ['X%i' % i for i in range(10)]

        D_train = pd.DataFrame(X_train, columns=column_names)
        D_test = pd.DataFrame(X_test, columns=column_names)

        lastochka = LastochkaTransformer(verbose=True, n_final=3, n_initial=10)
        log = LogisticRegression()

        pipe = Pipeline(steps=[
                ('lastochka', lastochka),
                ('log', log)])

        pipe.fit(D_train, y_train)
        X_w = lastochka.transform(D_train)
        X_wt = lastochka.transform(D_test)

        for variable in column_names:
            vt = lastochka.get_transformer(variable)
            acceptable_values = vt.optimizer_instance.bin_stats["woe_value"]
            real_values_train = X_w[variable].unique()
            real_values_test = X_wt[variable].unique()
            self.assertTrue(set(acceptable_values) == set(real_values_train))
            self.assertTrue((set(acceptable_values)) == set(real_values_test))

    def testNonExistentEstimator(self):
        vt = VectorTransformer(optimizer="fake_optimizer",
                               n_final=2, n_initial=10,
                               specials=[], verbose=False, name="test",
                               total_non_events=1, total_events=1)

        self.assertRaises(NotImplementedError, vt.fit,
                          X=np.random.normal(size=100),
                          y=np.random.randint(0, 2, size=100))

    def testStats(self):
        cases = pd.DataFrame([
            (1440, 141, 18000, 600, -1.07756),
            (2490, 138, 18000, 600, -0.50841),
            (4590, 149, 18000, 600, 0.02649),
            (5400, 124, 18000, 600, 0.37268),
            (4080, 48, 18000, 600, 1.04145)],
            columns=["non_events", "events", "total_non_events", "total_events", "expected_value"])

        for index, case in cases.iterrows():
            _y = np.concatenate([np.zeros(int(case["non_events"])), np.ones(int(case["events"]))])
            woe_value = calculate_stats(_y, case["total_non_events"], case["total_events"])[-1]
            self.assertAlmostEqual(case["expected_value"], woe_value, places=5)

    def testEmpty(self):
        _X = pd.DataFrame(columns=["X1,X2"])
        _y = np.array([])
        lastochka = LastochkaTransformer()
        self.assertRaises(ValueError, lastochka.fit, X=_X, y=_y)

    def testVerbose(self):
        N_SAMPLES = 100
        X, y = make_classification(n_samples=N_SAMPLES, n_features=5, n_informative=2, random_state=42)
        column_names = ['X%i' % i for i in range(5)]
        X_df = pd.DataFrame(X, columns=column_names)
        X_df["CAT"] = np.random.choice(list("ABCDF"), N_SAMPLES)

        lastochka = LastochkaTransformer(verbose=False, n_final=3, n_initial=10)

        lastochka.fit(X_df, y)

    def testDuplicates(self):
        X = np.concatenate([np.ones(200), np.random.normal(100, 1, size=50)])
        y = np.random.randint(0, 2, size=250)
        vt = VectorTransformer(n_initial=10, n_final=5,
                               optimizer="full-search", name="X1", verbose=False,
                               total_non_events=len(y) - y.sum(),
                               total_events=y.sum(), specials=[])
        with self.assertWarns(UserWarning):
            vt.fit(X, y)

    def testNonBinary(self):
        X = np.random.normal(0, 1, size=(200, 3))
        y = np.random.randint(0, 3, size=200)
        lastochka = LastochkaTransformer()
        self.assertRaises(ValueError, lastochka.fit, X=X, y=y)

    def testInputList(self):
        X = [[1, 2, 3], [4, 5, 6]]
        y = [0, 1]
        lastochka = LastochkaTransformer()
        self.assertRaises(TypeError, lastochka.fit, X=X, y=y)

    def testInputArray(self):
        X = np.random.normal(0, 1, size=(200, 3))
        y = np.random.randint(0, 2, size=200)
        lastochka = LastochkaTransformer()
        lastochka.fit(X, y)
        X_w = lastochka.transform(X)
        self.assertIsInstance(X_w, np.ndarray)

    def testIncrease(self):
        _boston = load_boston()
        X = pd.DataFrame(_boston["data"], columns=_boston["feature_names"])

        y = (_boston["target"] >= np.median(_boston["target"])).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

        lastochka = LastochkaTransformer(verbose=True, n_final=3, n_initial=10)
        log = LogisticRegression()
        slog = LogisticRegression()

        pipe = Pipeline(steps=[
                ('lastochka', lastochka),
                ('log', log)])

        pipe.fit(X_train, y_train)
        slog.fit(X_train, y_train)

        pipe_probas = pipe.predict_proba(X_test)[:, 1]
        slog_probas = slog.predict_proba(X_test)[:, 1]

        pipe_auc = roc_auc_score(y_test, pipe_probas)
        slog_auc = roc_auc_score(y_test, slog_probas)
        print("Pipe AUC: %0.5f, Log AUC: %0.5f" % (pipe_auc, slog_auc))
        self.assertGreater(pipe_auc, slog_auc)

    def testCategory(self):
        _boston = load_boston()
        X = pd.DataFrame(_boston["data"], columns=_boston["feature_names"])
        X["RAD_CAT"] = X["RAD"].astype(str)
        X = X.drop("RAD", axis=1)
        y = (_boston["target"] >= np.median(_boston["target"])).astype(int)
        lastochka = LastochkaTransformer(verbose=True, n_final=3, n_initial=10)
        lastochka.fit(X, y)
        lastochka.transform(X)
        optimizer_instance = lastochka.get_transformer("RAD_CAT").optimizer_instance
        self.assertIsInstance(optimizer_instance, CategoryOptimizer)

    def testMissing(self):
        _boston = load_boston()
        X = pd.DataFrame(_boston["data"], columns=_boston["feature_names"])
        np.random.seed(2)
        indexes = np.random.choice(X.index.tolist(), 200)
        X.loc[indexes, "ZN"] = np.nan
        y = (_boston["target"] >= np.median(_boston["target"])).astype(int)
        lastochka = LastochkaTransformer(verbose=True, n_final=3, n_initial=10)
        lastochka.fit(X, y)
        lastochka.transform(X)
        self.assertTrue(lastochka.get_transformer("ZN").missing_woe_value is not None)

    def testSpecial(self):
        _boston = load_boston()
        X = pd.DataFrame(_boston["data"], columns=_boston["feature_names"])
        y = (_boston["target"] >= np.median(_boston["target"])).astype(int)
        specs = {"PTRATIO": [20.2, 14.7]}
        lastochka = LastochkaTransformer(verbose=True, n_final=3, n_initial=10, specials=specs)
        lastochka.fit(X, y)
        lastochka.transform(X)
        self.assertTrue(list(lastochka.get_transformer("PTRATIO").specials_stats.keys()) == specs["PTRATIO"])


if __name__ == "__main__":
    unittest.main()
