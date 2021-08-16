# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:29:52 2021

@author: SONG
"""


import quandl as ql
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

def dataset_split(X_train, y_train, window_size):
    X = []
    y = []
    X_train = X_train.values
    y_train = y_train.values
    for i in range(len(X_train) - window_size - 1):
        X_ = X_train[i : i + window_size, :]
        y_ = y_train[i : i + window_size, ]
        X.append(X_)
        y.append(y_)
    return np.array(X), np.array(y)

def normalize(data):
    data_mean = data.mean(axis = 0)
    data_std = data.std(axis = 0)
    return (data - data_mean) / data_std

df = pd.read_csv("df.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.index = df["Date"]
df = df.drop(["Date"], axis = 1)

X = df.drop(["ret", "price"], axis = 1)
y = df["ret"] * 0.01
y = pd.DataFrame(y)
y.loc[y["ret"] >= 0] = 1
y.loc[y["ret"] < 0 ] = 0
y = pd.DataFrame(y, dtype = int)


# "blchs", "ntrat", "mirev", "cptra", "mktcp", "avbls"

X = df[["blchs", "ntrat", "mirev", "cptra", "mktcp"]]
X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, shuffle = False)

window_size = 180

X_train, y_train = dataset_split(X_train, y_train, window_size = window_size)
X_valid, y_valid = dataset_split(X_valid, y_valid, window_size= window_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation = "relu", input_shape = (X_train.shape[1], X_train.shape[2])), 
    tf.keras.layers.Conv1D(filters = 64, kernel_size = 3, activation = "relu"),  
    tf.keras.layers.Dropout(0.5), 
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(50, activation ="relu"),
    tf.keras.layers.Dense(1, activation = "softmax")
    ])

model.compile(optimizer='adam',
              loss= "binary_crossentropy", metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, 
                    validation_data=(X_valid, y_valid))




