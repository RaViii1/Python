from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
import pandas as pd
import numpy as np
# %% Zadanie 3.2
data = pd.read_excel('loan_data.xlsx')
columns = list(data.columns)
def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
qualitative_to_0_1(data, 'Gender', 'Female')
qualitative_to_0_1(data, 'Married', 'Yes')
qualitative_to_0_1(data, 'Self_Employed', 'Yes')
qualitative_to_0_1(data, 'Education', 'Graduate')
qualitative_to_0_1(data, 'Loan_Status', 'Y')
# %% Zadanie 3.3
def Metrics (tp, fp, tn, fn):
    sensivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    specifity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    if sensivity + precision != 0: F1 = (2 * sensivity * precision) / (sensivity + precision)
    else: F1 = 0
    print("Sensivity: " + str(sensivity))
    print("Precision: " + str(precision))
    print("Specifity: " + str(specifity))
    print("Accuracy: " + str(accuracy))
    print("F1: " + str(F1))
Metrics(7, 26, 17, 73)
Metrics(0, 33, 0, 90)
