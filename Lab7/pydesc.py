# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 20:22:47 2022

@author: Alicja
"""

from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
from keras.layers import Conv2D, Flatten,Dense, AveragePooling2D, MaxPooling2D


#7.1
train, test = mnist.load_data()
X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
#Dostosowanie danych
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

#Liczba epok
class_cnt = np.unique(y_train).shape[0] #zwrócenie unikalnych wartości i zmiana wielkości na 'n'

filter_cnt = 32 #liczba filtrów wyjściowych
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu' #funkcja aktywacji np: Step, Sigmoid, tanh, Relu , Elu, leaky Relu
kernel_size = (3,3) #wymiary 3x3 wysokość i szerokość okna splotu 2D

model = Sequential() #model sekwencyjny każda warstwa ma 1 tensor wejsciowy i jeden wyjsciowy
conv_rule = 'same' #padding "valid" or "same", "valid" means no padding, "same" results in padding
#Conv2D(32, (3,3), padding="same", activation="relu")
model.add(Conv2D(input_shape = X_train.shape[1:],
                 filters=filter_cnt,
                 kernel_size = kernel_size,
                 padding = conv_rule, 
                 activation = act_func))
model.add(Flatten())
model.add(Dense(class_cnt, activation='softmax'))
model.compile(optimizer=Adam(learning_rate),
              loss='SparseCategoricalCrossentropy',
              metrics='accuracy')

model.summary() #podsumowanie modelu
model.fit(x = X_train, y = y_train,epochs = class_cnt ,validation_data=(X_test, y_test))

#%%
#7.3

filter_cnt = 32
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'
kernel_size = (3,3)
pooling_size = (2,2) # używane do zmniejszenia wymiarów modelu 
#np: model.add(MaxPooling2D(pool_size=(2,2)))

model = Sequential()
conv_rule = 'same' # "valid" or "same", "valid" means no padding, "same" results in padding

model.add(Conv2D(input_shape = X_train.shape[1:],
                 filters=filter_cnt,
                 kernel_size = kernel_size,
                 padding = conv_rule, activation = act_func))
model.add(MaxPooling2D(pooling_size))
model.add(Flatten())
model.add(Dense(class_cnt, activation='softmax'))
model.compile(optimizer=Adam(learning_rate), 
              loss='SparseCategoricalCrossentropy',
              metrics='accuracy')

model.summary()
model.fit(x = X_train, y = y_train,
          epochs = class_cnt ,
          validation_data=(X_test, y_test))
