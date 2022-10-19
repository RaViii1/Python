#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:08:50 2022

@author: student
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
data = pd.read_excel('practice_lab_2.xlsx')


#zad1
korelacje = data.corr()
X =data.iloc[:,:data.shape[1]-1]
y = data.iloc[:,-1]

fig, ax = plt.subplot(X.shape[1], 1 ,figsize=(5,20))
for i, col in enumerate(X.colums):
    ax[i].scatter(X[col],y)
#zad2.2
def testuj_model(n):
    s = 0 #suma
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=221,shuffle=True)
        linReg = LinearRegression()
        linReg = LinearRegression()
        linReg.fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        s += mean_absolute_percentage_error (y_test, y_pred)
    return s/n

testuj_model(10)

#zad2.3
def usuniecie(n):
    s = 0
    for i in range(n):
      X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=221,shuffle=True) 
      outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
      X_train_no_outliers = X_train.loc[~outliers,:] ##usuwanie
      y_train_no_outliers = y_train[~outliers]
      linReg = LinearRegression()
      linReg.fit(X_train_no_outliers,y_train_no_outliers)
      y_pred = linReg.predict(X)
      s += mean_absolute_percentage_error (y_test, y_pred)
    return s/n      
usuniecie(10)    
def zmiana(n):
        s = 0
        for i in range(n):
          X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=221,shuffle=True) 
          outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
          X_train_no_outliers = X_train.loc[~outliers,:] ##usuwanie
          y_train_no_outliers = y_train[~outliers]
          linReg = LinearRegression()
          linReg.fit(X_train_no_outliers,y_train_no_outliers)
          y_pred = linReg.predict(X)
          s += mean_absolute_percentage_error (y_test, y_pred)
        return s/n      
from sklearn.datasets import load_diabetes
data2 = load_diabetes()   
dane = pd.DataFrame(data2.data, columns=data2.feature_names)
dane['target'] = data2.target
korelacje2 = dane.corr()

