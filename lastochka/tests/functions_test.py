import numpy as np
import pandas as pd
from lastochka.core.functions import generate_combs,make_edges,calculate_loc_woe
from lastochka.core.optimizer import WingOptimizer
from lastochka import WingsOfEvidence
import logging

if __name__ == "__main__":
    T_SIZE = 200
    X = np.random.uniform(0, 1, size=T_SIZE)
    y = np.random.randint(0, 2, size=T_SIZE)
    wo = WingOptimizer(X, y, total_good=y.sum(), total_bad=len(y)-y.sum(), n_initial=10, n_target=5,
                       optimizer="adaptive")
    #init_edges = wo._initial_split()
    wo.optimize()