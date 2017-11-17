# -*- coding: utf-8 -*-

import sys
sys.path.insert(0,'..')
import pandas as pd
import datetime as dt
from wing import WingsOfEvidence
from sklearn.datasets import make_classification
if __name__ == "__main__":
    N_SAMPLES = 20000
    N_FEATURES = 5
    X,y = make_classification(n_samples=N_SAMPLES,n_features=N_FEATURES)
    X_df = pd.DataFrame(X,columns=["X_%i"%i for i in range(N_FEATURES)])
    colnames = X_df.columns.tolist()
    adapt_st = dt.datetime.now()
    wing = WingsOfEvidence(n_initial=10, n_target=5,
                            columns_to_apply="all",
                            optimizer="adaptive",
                            #optimizer="full-search",
                            #mass_spec_values={"Age": {20: "ZERO"}}
                            )
    wing.fit(X_df, y)
    adapt_fn = dt.datetime.now()
    adapt_dt = adapt_fn - adapt_st
    fullsr_st = dt.datetime.now()
    wing = WingsOfEvidence(n_initial=10, n_target=5,
                            columns_to_apply=colnames,
                            #optimizer="adaptive",
                            optimizer="full-search",
                            #mass_spec_values={"Age": {20: "ZERO"}}
                            )
    wing.fit(X_df, y)
    fullsr_fn = dt.datetime.now()
    fullsr_dt = fullsr_fn - fullsr_st

    print("Adaptive timing: %s"%(adapt_dt))
    print("###" * 20)
    print("###" * 20)
    print("###" * 20)
    print("Fullsearch timing: %s"%(fullsr_dt))
