# тест создан для проверки woe-разбиения на монотонность.
import sys
sys.path.insert(0,'..')
import pandas as pd
import datetime as dt
from lastochka import WingsOfEvidence
from sklearn.datasets import make_classification
import logging
if __name__ == "__main__":
    N_SAMPLES = 20000
    N_FEATURES = 5
    X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES,random_state=42)
    X_df = pd.DataFrame(X, columns=["X_%i" % i for i in range(N_FEATURES)])
    #colnames = ["X_0"]
    wings = WingsOfEvidence(n_initial=10, n_target=5,
                           columns_to_apply="all",
                           optimizer="full-search",
                           # mass_spec_values={"Age": {20: "ZERO"}}
                           )
    wings.fit(X_df,y)
    for feature in X_df.columns:
        loc_w = wings.fitted_wing[feature]
        print(loc_w.get_wing_agg())