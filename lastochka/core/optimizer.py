# -*- coding: utf-8 -*-
# TODO: Refactor optimizers to classes

from .functions import make_edges, generate_combs, check_variant, add_infinity, split_by_edges
from sklearn.base import BaseEstimator, TransformerMixin


class FullSearchOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self, n_initial: int, n_final: int):
        self.n_initial = n_initial
        self.n_final = n_final
        self.edges = None
        self.gini = None

    def fit(self, X, y):
        initial_edges = make_edges(X, self.n_initial)
        all_edge_variants = generate_combs(initial_edges[1:-1], self.n_final)
        mono_variants = []

        for edge_variant in all_edge_variants:
            edge_variant = add_infinity(edge_variant)
            bins = split_by_edges(X, edge_variant)
            is_mono, gini = check_variant(bins, y,
                                          total_good=len(y) - sum(y),
                                          total_bad=sum(y))
            if is_mono:
                mono_variants.append((edge_variant, gini))

        self.edges, self.gini = sorted(mono_variants, key=lambda x: x[1])[-1]

        return self

    def transform(self, X, y=None):

        return X


class CategoryOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        return X


OPTIMIZERS = {
    "full-search": FullSearchOptimizer,
    "category": CategoryOptimizer
}
