# -*- coding: utf-8 -*-
# this is hack to import from pre-parent directory
import sys
import unittest

sys.path.insert(0,'..')
import pandas as pd
from lastochka import WingsOfEvidence

class BaseTest(unittest.TestCase):
    """
    Класс для тестов 
    """
    def test_columns(self):
        """
        тестируем функционал фильтрации колонок
        """
        cols = list("ABCD")
        wings = WingsOfEvidence(columns_to_apply=cols)
        self.assertEqual(cols,wings.columns_to_apply)
    def testAllTypes(self):
        train_df = pd.read_csv("../datasets/titanic/train.csv", sep=",")
        columns = [ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
        X_train = train_df[columns]
        y_train = train_df["Survived"]
        wings = WingsOfEvidence(n_initial=10,n_target=5,
                                columns_to_apply="all",
                                #optimizer="adaptive"
                                optimizer="full-search",
                                mass_spec_values={"Age":{20:"ZERO"}}
                                )
        wings.fit(X_train,y_train)
        X_tsrf = wings.transform(X_train)
    def testTitanicC(self):
        """
        тестируем числовое woe
        """
        train_df = pd.read_csv("../datasets/titanic/train.csv",sep=",")
        colnames = ["Age"]
        X_train = train_df[colnames]
        y_train = train_df["Survived"]
        wings = WingsOfEvidence(n_initial=10,n_target=5,
                                columns_to_apply=colnames,
                                #optimizer="adaptive"
                                optimizer="full-search",
                                mass_spec_values={"Age":{20:"ZERO"}}
                                )
        wings.fit(X_train,y_train)
        X_tsrf = wings.transform(X_train)
        #print(X_tsrf)
    def testTitanicCSpec(self):
        """
        тестируем случай, когда задано спец-значение, а его нет в обучающей выборке
        """
        train_df = pd.read_csv("../datasets/titanic/train.csv",sep=",")
        colnames = ["Age"]
        X_train = train_df[colnames]
        y_train = train_df["Survived"]
        wings = WingsOfEvidence(n_initial=12,n_target=5,
                                columns_to_apply=colnames,
                                #optimizer="adaptive",
                                optimizer="full-search",
                                mass_spec_values={"Age":{0:"ZERO"}}
                                )
        wings.fit(X_train,y_train)
        loc_w = wings.fitted_wing["Age"]
        result = ["%0.5f"%el for el in loc_w.transform(pd.Series([0,0,0]))["woe"].values]
        check_w = loc_w.get_wing_agg()["woe"].min()
        tester = ["%0.5f"%check_w for i in range(3)]
        self.assertEqual(result,tester)
    def testOnlyValues(self):
        """
        Тестируем случай, когда задан параметр only_values=False
        """
        train_df = pd.read_csv("../datasets/titanic/train.csv",sep=",")
        colnames = ["Age"]
        X_train = train_df[colnames]
        y_train = train_df["Survived"]
        wings = WingsOfEvidence(n_initial=20,n_target=5,
                                columns_to_apply=colnames,
                                #optimizer="adaptive",
                                optimizer="full-search",
                                mass_spec_values={"Age":{20:"ZERO"}},
                                only_values=False
                                )
        wings.fit(X_train,y_train)
        X_tsrf = wings.transform(X_train)
        test_names = ["WOE_Age","WOE_g_Age"]
        self.assertEqual(X_tsrf.columns.tolist(),test_names)
    def checkMono(self):
        train_df = pd.read_csv("../datasets/titanic/train.csv",sep=",")
        colnames = ["Age"]
        X_train = train_df[colnames]
        y_train = train_df["Survived"]
        wings = WingsOfEvidence(n_initial=10,n_target=5,
                                columns_to_apply=colnames,
                                #optimizer="adaptive",
                                optimizer="full-search",
                                mass_spec_values={"Age":{0:"ZERO"}}
                                )
        wings.fit(X_train,y_train)
        loc_w = wings.fitted_wing["Age"]
        does_local = loc_w._check_mono(loc_w.get_wing_agg()["woe"])
        self.assertEqual(does_local,True)

if __name__ == "__main__":
    unittest.main()