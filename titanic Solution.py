# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 21:28:38 2019

@author: Vikash Sarraf
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset= pd.read_csv('titanic_data.csv')
sns.heatmap(dataset.isnull(),cmap='viridis',yticklabels=False,cbar=False)
sns.set_style('whitegrid')
sns.boxplot(x='Pclass',y='Age',data=dataset)

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
dataset['Age']=dataset[['Age','Pclass']].apply(impute_age,axis=1)
dataset.drop('Cabin',axis=1,inplace=True)
dataset.dropna(inplace=True)
sex=pd.get_dummies(dataset['Sex'],drop_first=True)
embarked=pd.get_dummies(dataset['Embarked'],drop_first=True)
dataset.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
dataset=pd.concat([dataset,sex,embarked],axis=1)
x=dataset.drop('Survived',axis=1)
y=dataset['Survived']
from sklearn.cross_validation import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(xtrain,ytrain)
ypred=log.predict(xtest)
print('Accuracy',log.score(xtest,ytest))

from sklearn.metrics import confusion_matrix,classification_report
print('cf',confusion_matrix(ytest,ypred))
print('cr',classification_report(ytest,ypred))
