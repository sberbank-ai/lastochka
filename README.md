# Lastochka
[![Build Status](https://travis-ci.org/sberbank-ai/lastochka.svg?branch=master)](https://travis-ci.org/sberbank-ai/lastochka)
[![codecov](https://codecov.io/gh/sberbank-ai/lastochka/branch/master/graph/badge.svg)](https://codecov.io/gh/sberbank-ai/lastochka)
[![PyPI version](https://badge.fury.io/py/lastochka.svg)](https://badge.fury.io/py/lastochka)

Weight of Evidence transformation, implemented in Python. 

# Quickstart

1. Install the package:
```bash
pip install lastochka
```

2. Use module as scikit-learn transformer:
```python
import pandas as pd
from lastochka import LastochkaTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, n_features=10, n_informative=2, random_state=42)

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

test_proba = pipe.predict_proba(D_test)
```


