
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from .optimizer import OPTIMIZERS
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import type_of_target
from warnings import warn
from typing import Union, Tuple,Dict


class VectorTransformer(BaseEstimator, TransformerMixin):
    """
    Class for WoE calculation and optimization per one feature
    """
    def __init__(self, n_initial: int, n_final: int, optimizer: str, specials: Dict, verbose: bool, name: str):
        """
        Initialize transformer for one feature
        :param n_initial:
        :param n_final:
        :param optimizer:
        :param specials:
        :param verbose
        """
        self.optimizer = optimizer
        self.n_initial = n_initial
        self.n_final = n_final
        self.specials = specials
        self.verbose = verbose
        self.name = name
        self.optimizer_class = None
        self.optimizer_instance = None

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def _preprocess_specials(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess special values
        :param X:
        :param y:
        :return:
        """
        non_missing_mask = np.isfinite(X)
        X = X[non_missing_mask]
        y = y[non_missing_mask]

        return X, y

    def _preprocess_missing(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess missing values
        :param X:
        :param y:
        :return:
        """
        for key, special_value in self.specials.items():
            _mask = (X == special_value)
            X = X[~_mask]
            y = y[~_mask]

        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit one feature
        :param X:
        :param y:
        :return:
        """
        print("Processing variable: %s" % self.name)
        self._print("Input dataset before preprocessing : %i" % len(X))
        X, y = self._preprocess_specials(X, y)
        self._print("Input dataset after specials       : %i" % len(X))
        X, y = self._preprocess_missing(X, y)
        self._print("Input dataset after missing        : %i" % len(X))

        if X.dtype != np.dtype("O"):
            if np.unique(X).shape[0] <= (len(X) / self.n_initial):
                self.optimizer_class = OPTIMIZERS.get("category")
                warn("Too low amount of unique values - category optimizer will be used")
            else:
                self.optimizer_class = OPTIMIZERS.get(self.optimizer)
        else:
            self.optimizer_class = OPTIMIZERS.get("category")

        if not self.optimizer_class:
            raise NotImplementedError("Optimizer %s is not yet implemented, allowed optimizers are: %s" %
                                      (self.optimizer, ', '.join(OPTIMIZERS.keys())))

        self.optimizer_instance = self.optimizer_class(self.n_initial, self.n_final).fit(X, y)

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        """

        :param X:
        :param y:
        :return:
        """
        return self.optimizer_instance.transform(X)


class LastochkaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_initial: int = 10,
                 n_final: int = 5,
                 optimizer: str = "full-search",
                 verbose: bool = False,
                 specials: Dict = None):
        """
        Performs the Weight Of Evidence transformation over the input X features using information from y vector.
        :param n_initial: Initial amount of quantile-based groups
        :param n_final: Amount of maximum groups after the groups construction
        :param optimizer: Optimizer string name
        :param verbose: boolean flag to add verbose output
        """
        self.n_initial = n_initial
        self.n_final = n_final
        self.optimizer = optimizer
        self.verbose = verbose
        self.specials = specials if specials else {}
        self.transformers = {}
        self.feature_names = []

    def fit(self,
            X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]):
        """
        Fits the input data
        :param X: data matrix
        :param y: target vector
        :return: self
        """

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
        elif isinstance(X, np.ndarray):
            self.feature_names = ["X%i" % i for i in range(X.shape[-1])]
        else:
            raise TypeError("X vector is not np array neither data frame")

        X, y = self._check_inputs(X, y)

        for index, vector in enumerate(np.nditer(X, flags=['external_loop'], order='F')):
            vector_specials = self.specials.get(self.feature_names[index], {})

            vector_transformer = VectorTransformer(self.n_initial,
                                                   self.n_final,
                                                   self.optimizer,
                                                   specials=vector_specials,
                                                   verbose=self.verbose,
                                                   name=self.feature_names[index])

            vector_transformer.fit(vector, y)
            self.transformers[self.feature_names[index]] = vector_transformer

        return self

    def _check_inputs(self,
                      X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check input data
        :param X:
        :param y:
        :return:
        """
        if type_of_target(y) != "binary":
            raise ValueError("y vector should be binary")

        X, y = check_X_y(X, y, accept_sparse=False, force_all_finite=False, dtype=None, y_numeric=True)
        return X, y

    def transform(self,
                  X: Union[pd.DataFrame, np.ndarray],
                  y: Union[pd.Series, np.ndarray] = None):
        """
        Checks and transforms input arrays
        :param X: X data array
        :param y: target array
        :return: transformed data
        """
        X = check_array(X, accept_sparse=False, force_all_finite=False, dtype=None)

        X_w = np.zeros(X.shape)

        for index, vector in enumerate(np.nditer(X, flags=['external_loop'], order='F')):
            X_w[:, index] = self.transformers[self.feature_names[index]].transform(vector)

        if isinstance(X, pd.DataFrame):
            X_w = pd.DataFrame(X_w, columns=self.feature_names)

        return X_w
