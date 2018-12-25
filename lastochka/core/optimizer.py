# -*- coding: utf-8 -*-
# TODO: Refactor optimizers to classes

from .functions import make_edges, generate_combs, check_variant, add_infinity, split_by_edges


class WingOptimizer:
    def __init__(self, X, y, total_good, total_bad, n_initial, n_target, optimizer="adaptive", verbose=False):
        """
        :param X (np.ndarray):
            Одномерный X-вектор для поиска перестановок
        :param y (np.ndarray):
            Одномерный y-вектор для поиска перестановок
        :param n_initial (int):
            С какого значения инициируем разбиение
        :param n_target (int):
            каков размер макс. групп
        :param optimizer (str):
            Тип оптимизатора
        """
        self.X = X
        self.y = y
        self.n_initial = n_initial
        self.n_target = n_target
        self.optimizer = optimizer
        self.total_good = total_good
        self.total_bad = total_bad
        self.verbose = verbose
        self.init_edges = None

    def optimize(self):
        """
        Класс инициирует основную логику.
        :return:
         opt_edges:
            Оптимально разбитые границы
        """
        self.init_edges = self._initial_split()
        optimization_result = self._search_optimals()
        return optimization_result

    def _initial_split(self):
        """
        Рассчитывает инициирующие границы
        """
        return make_edges(self.X, self.n_initial)

    def _search_optimals(self):
        """

        :return:
            (opt_edges,gini):
                Возвращает оптимальные границы и значение gini на лучшей разбивке
        """
        if self.optimizer == "full-search":
            print("Doing full-search with init: %s" % self.init_edges)
            all_edge_variants = generate_combs(self.init_edges[1:-1], self.n_target)
            print("FS variants total %i" % len(all_edge_variants))
            mono_variants = []
            for edge_variant in all_edge_variants:
                edge_variant = add_infinity(edge_variant)
                bins = split_by_edges(self.X, edge_variant)
                is_mono, gini = check_variant(bins, self.y,
                                              total_good=self.total_good,
                                              total_bad=self.total_bad)
                if is_mono:
                    mono_variants.append((edge_variant, gini))
            print("Total mono variants: %i" % len(mono_variants))
            optimization_result = sorted(mono_variants, key=lambda x: x[1])[-1]

        else:
            raise NotImplementedError("Algorithm %s is not implemented" % self.optimizer)

        return optimization_result
