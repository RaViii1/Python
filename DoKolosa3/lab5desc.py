from sklearn.datasets import load_digits
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

#ZAD-5.2

#Wczytujemy dane
data = load_digits()
X = data.data
y = data.target

#Zamiana na zmienna kategorialna
y = pd.Categorical(y)
y = pd.get_dummies(y)

#Jakies wykresy pani Pluty, raczej do wyjebania
plt.gray()
plt.matshow(data.images[515])

#Dzielimy na zbior testowy i treningowy
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

X_val = X_train[:400]
y_val = y_train[:400]
X_train = X_train[401:]
y_train = y_train[401:]

model = Sequential()#Model sekwencyjny, tzn wartswy ida jedna po drugiej

#Dodawanie warstw gestych ( kazdy neuron kazdej warstwy gestej jest polaczony z neuronem warstwy poprzedniej)
#(ilosc neuronow, liczba cech wejscia jako rozmiar wejscia ( tylko pierwsza warstwa ), activation = 'funkcja aktywacji')
model.add(layers.Dense(64, input_shape = (X_train.shape[1],), activation = 'relu' ))
model.add(layers.Dense(64, activation=('relu')))

#W Ostatniej warstwie w przypadku klasyfikacji wieloklasowej zastosowana musi byc funkcja softmax
model.add(layers.Dense(y_train.shape[1], activation = 'softmax'))

model.summary()#Summary wyswietla podsumowanie utworzonego modelu

#(Rodzaj optymalizatora Adam -  czasem podaje sie w nawiasie learning_rate, funkcja strat, metryka jakosci pracy sieci)
model.compile(optimizer= 'Adam',loss = 'categorical_crossentropy',metrics = 'acc')

#batch_size - porcje, epochs - ilosc epok, jesli nie chcemy wiadomosci o kazdej epoce ustawiamy verbose na 0 
history = model.fit(X_train, y_train, batch_size = 8, epochs=5, validation_data = (X_val, y_val),verbose = 1)

#Predykcja
y_pred = model.predict(X_test)

#Pobranie historii uczenia modelu oraz jej wizualizacja
floss_train = history.history['loss']
floss_test = history.history['val_loss']
acc_train = history.history['acc']
acc_test = history.history['val_acc']

fig,ax = plt.subplots(1,2, figsize=(20,10))
epochs = np.arange(0, 5)

#Wykres funkcji strat
ax[0].plot(epochs, floss_train, label = 'floss_train')
ax[0].plot(epochs, floss_test, label = 'floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()

#Wykres dokladnosci
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label = 'acc_train')
ax[1].plot(epochs, acc_test, label = 'acc_test')
ax[1].legend()

