import pandas as pd
from wing import WingsOfEvidence

train_df = pd.read_csv("../datasets/titanic/train.csv", sep=",")
columns = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
X_train = train_df[columns]
y_train = train_df["Survived"]
wings = WingsOfEvidence(n_initial=10, n_target=5,
                        columns_to_apply="all",
                        # optimizer="adaptive"
                        optimizer="full-search",
                        mass_spec_values={"Age": {20: "ZERO"}}
                        )
wings.fit(X_train, y_train)
X_tsrf = wings.transform(X_train)
print(X_tsrf)