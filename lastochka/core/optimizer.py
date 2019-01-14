from .functions import generate_combs, calculate_overall_stats, add_infinity, check_mono, gini_index
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from warnings import warn
from tqdm import tqdm
import sys
import numpy as np


class FullSearchOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 name: str,
                 n_initial: int,
                 n_final: int,
                 total_events: int,
                 total_non_events: int,
                 verbose: bool):
        self.name = name
        self.n_initial = n_initial
        self.n_final = n_final
        self.total_events = total_events
        self.total_non_events = total_non_events
        self.verbose = verbose

        self.edges = None
        self.gini = None
        self.bin_stats = None

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def fit(self, X, y):
        _, initial_edges = pd.qcut(X, self.n_initial, duplicates="drop", retbins=True)

        if len(initial_edges) != self.n_initial + 1:
            warn("Dataset contains too many duplicates, "
                 "amount of groups on initial stage was set to: %i" % len(initial_edges))

        all_edge_variants = generate_combs(initial_edges[1:-1], self.n_final, len(initial_edges)+1)

        mono_variants = []
        if self.verbose:
            edges_iterator = tqdm(all_edge_variants, desc="Variable %s optimization" % self.name, file=sys.stdout)
        else:
            edges_iterator = all_edge_variants

        for edge_variant in edges_iterator:
            edge_variant = add_infinity(edge_variant)
            X_b = np.digitize(X, edge_variant)
            bin_stats = calculate_overall_stats(X_b, y,
                                                total_events=self.total_events,
                                                total_non_events=self.total_non_events)
            if check_mono(bin_stats.woe_value):
                bin_stats = bin_stats.sort_values(by="local_event_rate", ascending=False)
                gini = gini_index(bin_stats.events.values, bin_stats.non_events.values)
                mono_variants.append((edge_variant, gini))

        if len(mono_variants) == 0:
            warn("No monotonic bins combination found, initial split will be used")
            self.edges = add_infinity(initial_edges[1:-1])
            self.gini = None
        else:
            self.edges, self.gini = sorted(mono_variants, key=lambda x: x[1])[-1]
            self._print("Best variant gini: %0.5f" % self.gini)

        X_b = np.digitize(X, self.edges)
        self.bin_stats = calculate_overall_stats(X_b, y,
                                                 total_events=self.total_events,
                                                 total_non_events=self.total_non_events)

        return self

    def transform(self, X, y=None) -> np.ndarray:
        X_b = pd.DataFrame(np.digitize(X, self.edges), columns=["bin_id"])
        X_w = pd.merge(X_b, self.bin_stats[["woe_value"]],
                       how="left", left_on="bin_id", right_index=True)["woe_value"].values
        return X_w


class CategoryOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 name: str,
                 n_initial: int,
                 n_final: int,
                 total_events: int,
                 total_non_events: int,
                 verbose: bool):
        self.name = name
        self.n_initial = n_initial
        self.n_final = n_final
        self.total_events = total_events
        self.total_non_events = total_non_events
        self.verbose = verbose

        self.edges = None
        self.gini = None
        self.bin_stats = None

    def _print(self, msg: str):
        if self.verbose:
            print(msg)

    def fit(self, X, y):
        bin_stats = calculate_overall_stats(X, y,
                                            total_events=self.total_events,
                                            total_non_events=self.total_non_events)
        self.bin_stats = bin_stats
        return self

    def transform(self, X, y=None) -> np.ndarray:
        X_b = pd.DataFrame(X, columns=["bin_id"])
        X_w = pd.merge(X_b, self.bin_stats[["woe_value"]],
                       how="left", left_on="bin_id", right_index=True)["woe_value"]
        X_w = X_w.fillna(self.bin_stats["woe_value"].max())
        return X_w


OPTIMIZERS = {
    "full-search": FullSearchOptimizer,
    "category": CategoryOptimizer
}
