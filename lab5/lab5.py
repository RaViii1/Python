from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt
from keras import layers
from keras.utils import plot_model
import random
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

data = load_digits()
X = data.data
y = data.target

y = pd.Categorical(y)
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_val = X_train[:400]
y_val = y_train[:400]
X_train = X_train[400:]
y_train = y_train[400:]

model = Sequential()
model.add(Dense(64, input_shape = (X_train.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.summary()
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=('accuracy'))

history = model.fit(X_train, y_train, batch_size=8, epochs=5, verbose=1, validation_data=(X_val, y_val)).history
model.predict(X_test)

floss_train = history['loss']
floss_test = history['val_loss']
acc_train = history['accuracy']
acc_test = history['val_accuracy']

fig,ax = plt.subplots(1,2, figsize=(20,10))
epochs = np.arange(0, 5)

ax[0].plot(epochs, floss_train, label = 'floss_train')
ax[0].plot(epochs, floss_test, label = 'floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label = 'acc_train')
ax[1].plot(epochs, acc_test, label = 'acc_test')
ax[1].legend()
# %%
