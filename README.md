# Оглавление  
1. [О проекте] (#О-проекте)  
2. [Как использовать?] (#Как-использовать)  
3. [Как улучшить?] (#Как-улучшить)  
4. [Контакты] (#Контакты)  

## О проекте  
Данный проект реализует sklearn-based Transformer для Weight of evidence преобразования.  
Это одно из наиболее удобных и результативных преобразований для логистической регрессии.
Детальная документация так же доступна в SAS Miner -> Credit Scoring -> Interactive Grouping  

## Как использовать?  

1. Загрузите себе репо:  
```
git clone https://sbt-gitlab.ca.sbrf.ru/Trusov-IA/wing.git
```
2. Установите модуль:  
```
cd wing
python setup.py install
```
3. Импортируйте объект и работайте с ним как sklearn Transformer:  
```
from wing import WingsOfEvidence
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
cols = ["A","B","C"]
wings = WingsOfEvidence(n_initial=10,n_target=5,columns_to_apply=cols,only_values=True,optimizer="full-search")
log_reg = LogisticRegression(fit_intercept=True)
pipe = Pipeline([
        ("wings",wings),
        ("log",log_reg)
        ])
pipe.fit(X,y)
```

# Как улучшить? 
1. Загрузите себе репо:  
```
git clone https://sbt-gitlab.ca.sbrf.ru/Trusov-IA/wing.git
```
2. Сделайте свою ветку:  
```
git checkout -b issue-1
git add --all
git commit -m "Initial commit"
git push --set-upstream issue-1
```
3. Работайте с кодом локально, проверьте тесты в /tests. Когда все готово, делайте push:  
```
git add --all
git commit -m "Final commit"
git push
```
4. Сделайте в webUI GitLab Merge Request  
5. PROFIT!!!

# Контакты  
Трусов Иван - IATrusov@sberbank.ru