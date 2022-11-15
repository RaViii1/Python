# %%1 Import
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as SVM
import pandas as pd
import numpy as np
data = pd.read_excel('loan_data.xlsx')
columns = list(data.columns)

#metoda do zamiany danych texctowych na 1 - 0
def tekst_na_bin(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
tekst_na_bin(data, 'Gender', 'Female')
tekst_na_bin(data, 'Gender', 'Female')
tekst_na_bin(data, 'Married', 'Yes')
tekst_na_bin(data, 'Self_Employed', 'Yes')
tekst_na_bin(data, 'Education', 'Graduate')
tekst_na_bin(data, 'Loan_Status', 'Y')

cat_feature = pd.Categorical(data.Property_Area) #Z pól posiadających więcej niz 2 wartosci textowe
one_hot = pd.get_dummies(cat_feature) # zamieniane są na wartosci 0 1
data = pd.concat([data, one_hot], axis=1) # dodanie ich do danych
data = data.drop(columns=['Property_Area']) # usunięcie oryginalnej kolumny

vals = data.values.astype(np.float64)
y = data['Loan_Status'].values #kol wynikowa
X = data.drop(columns = ['Loan_Status']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9) #90% danych w zbiorze treningowym
y_train = np.array(y_train).astype('float') #?
y_test = np.array(y_test).astype('float') #?

#wpływ skalera na jakosc kalsyfikacji
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
#metoda najbliżyszych sąsiadów
models = [SVM(kernel='sigmoid'), kNN(weights='distance')]
for model in models:
    model.fit(np.array(X_train), y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

#%%2
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
data = pd.read_csv('voice_extracted_features.csv', sep=',')
columns = list(data.columns)
#Podziała na dane
def qualitative_to_0_1(data, column, value_to_be_1):
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
qualitative_to_0_1(data, 'label', 'female')
vals = data.values
X = vals[:,:-1]
y = vals[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # dopasowania, jak i transformacja
y_train = np.array(y_train).astype('float')
y_test = np.array(y_test).astype('float')
X_paced=PCA(2).fit_transform(X_train) #zostają 2 główne składowe
females = y_train == 1 #?
#Generowanie wykresu
fig,ax=plt.subplots(1,1)
ax.scatter(X_paced[females,0], X_paced[females,1], label='female')
ax.scatter(X_paced[~females,0], X_paced[~females,1], label='male')
ax.legend()
#Tworzenie obiektu pipeline
from sklearn.pipeline import Pipeline
pipe = Pipeline([
 ['transformer', PCA(7)],
 ['scaler', MinMaxScaler()],
 ['classifier', kNN(weights='distance')]])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
confusion_matrix(y_test, y_pred)

# %%
