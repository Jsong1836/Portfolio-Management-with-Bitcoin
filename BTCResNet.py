# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:20:31 2021

@author: SONG
"""

import numpy as np 
import pandas as pd
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

X = df[["blchs", "ntrat", "mirev", "cptra", "mktcp", "avbls"]]
X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = False)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, shuffle = False)

window_size = 100

X_train, y_train = dataset_split(X_train, y_train, window_size = window_size)
X_valid, y_valid = dataset_split(X_valid, y_valid, window_size= window_size)


inputs = tf.keras.Input(shape=(24, 24, 3))
x = tf.keras.layers.Conv1D(32, 3, activation='relu')(inputs)
x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
x = tf.keras.layers.MaxPooling1D(3)(x)

num_res_net_blocks = 10

def res_net_block(input_data, filters, conv_size):
    x = tf.keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, input_data])
    x = tf.keras.layers.Activation('relu')(x)
    return x
for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    res_net_model = tf.keras.Model(inputs, outputs)
    
    
    
