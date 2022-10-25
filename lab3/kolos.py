

import numpy as np
import pandas as pd

data = pd.read_excel("practise_lab_1.xlsx")
col = data.columns #a
vals = data.values #b

mean_col = np.mean(vals, axis=0) #c
mean_std = np.std(vals) #d
difference = vals - mean_std #e
max_row_val = np.max(vals, axis=1) #f
arr2 = vals * 2 #g

col_np = np.array(col) #h
col_max = col_np[np.max(vals) == np.max(vals, axis=0)] #h
arr9 = (vals < mean_std).sum(axis=0) #i

#zad2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_excel("practise_lab_1.xlsx")
col = data.columns
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
cor = data.corr()

fig, ax = plt.subplots(X.shape[1], 1, figsize=(5,20))
for i, col in enumerate(X.columns):
    ax[i].scatter(X[col], y)

def testuj (n):
    s = 0
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=221, shuffle=True)
        linreg = LinearRegression()
        linreg.fit(X_train, y_train)
        y_pred = linreg.predict(X_test)
        s += mean_absolute_error(y_test, y_pred)
    return s/n
testuj(50)