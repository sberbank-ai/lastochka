import numpy as np
from lastochka.core.optimizer import WingOptimizer

if __name__ == "__main__":
    T_SIZE = 200
    X = np.random.uniform(0, 1, size=T_SIZE)
    y = np.random.randint(0, 2, size=T_SIZE)
    wo = WingOptimizer(X, y, total_good=y.sum(), total_bad=len(y)-y.sum(), n_initial=10, n_target=5,
                       optimizer="full-search")
    wo.optimize()
