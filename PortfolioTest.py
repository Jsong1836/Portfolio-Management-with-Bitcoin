# -*- coding: utf-8 -*-
"""
Created on Mon May 31 16:01:02 2021

@author: SONG
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split



def get_equity(ticker, start_date = None, end_date = None, normalize = False):
    import yfinance 
    import datetime
    if start_date:
        start_date = start_date
    else:
        start_date = "1800-01-01"
    if end_date:
        end_date = end_date
    else: 
        end_date = datetime.date.today()
    df = yfinance.download(ticker, start_date = start_date, end_date = end_date)
    df.dropna(inplace = True)
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    if normalize is True:
        normalized = scaler.fit_transform(df, shuffle = False)
        return normalized
    else:
        return df
    return df
def normalize(data):
    data_mean = data.mean(axis = 0)
    data_std = data.std(axis = 0)
    return (data - data_mean) / data_std

def portfolio():
    lists = ["CL=F", "^IXIC", "^TNX", "GC=F", 
             "EURUSD=X", "JPY=X", "^RUT", "GBPUSD=X", "^N225"]
    data = get_equity(lists, start_date = "2017-03-01", end_date = "2019-12-31")
    data = data["Close"]
    return data

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

def returns(data):
    dataframe = np.log(1 + data.pct_change())
    dataframe.dropna(inplace = True)
    return dataframe

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

window_size = 200

X_train, y_train = dataset_split(X_train, y_train, window_size = window_size)
X_valid, y_valid = dataset_split(X_valid, y_valid, window_size= window_size)

model= tf.keras.models.Sequential([
    tf.keras.layers.Dense(250, activation = "relu", input_shape = (X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(125, activation = "relu"),
    tf.keras.layers.Dense(1)
    ])

model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs = 200, validation_data=(X_valid, y_valid))

pred = model.predict(X_test)
real_pred = tf.round(tf.nn.sigmoid(pred))

signal = np.array(real_pred)

Port = portfolio()
port = returns(Port)

raw_data = port.copy()
# Stats descriptive
raw_data = raw_data.loc["2013-03-14": "2019-12-31"]


concat = pd.concat([port, df["ret"]], axis = 1)
concat = concat.loc["2017-03-14": "2019-12-31"]
vanille = concat.copy()
vanille.ret = vanille.ret * 0.01
cum_vanille = vanille.apply(np.cumsum, axis = 0)
normal = cum_vanille.fillna(method = "bfill")
normal.dropna(inplace = True)
normal = normalize(normal)


concat.ret = concat.ret * 0.01
sans = concat.copy()
concat.ret = concat.ret * signal.flatten()

concat = concat.fillna(method = "bfill")
concat = concat.fillna(method = "pad")

avec = concat


mean_col = np.matrix(concat.mean(axis = 0))



w = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
w1 = np.array([[0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0]])

corr = np.array(concat.corr(method = "pearson"))

print("Returns with BTC", w * mean_col.T)
print("Variance with BTC Portfolio", np.sqrt(w * np.matrix(corr) * w.T))
print("Returns without BTC", w1 * mean_col.T)
print("Variance without BTC Portfolio", np.sqrt(w1 * np.matrix(corr) * w1.T))

vanille = vanille.fillna(method = "bfill")
vanille = vanille.fillna(method = "pad")
vanille.ret = vanille.ret * 0.01

print("Vanilla Mean:", np.matrix(vanille.mean(axis = 0)) * w.T)
print("Vanilla Std:", np.sqrt(w * np.matrix(vanille.corr()) * w.T))

def sharp(returns, risk_free, std):
    return (returns - risk_free) / std
        

# normal.plot(figsize = (15, 7), grid = True).legend(loc = "lower left")

sans = sans.fillna(method = "bfill")
sans = sans.fillna(method = "pad")

x  = [(np.matrix(sans))[i, :] * w.T for i in range(1023)] 
y = [(np.matrix(sans))[i, :] * w1.T for i in range(1023)]
x = np.array(x)
y = np.array(y)
x = x.flatten()
y = y.flatten()
k = np.stack((x, y), axis = 1)

mean = pd.DataFrame(k, index = sans.index, columns= ["Avec_BTC", "Sans_BTC"])
mean_cum = mean.apply(np.cumsum, axis = 0)

mean_cum.plot(figsize = (10, 5), grid = True).legend(loc = "lower left")

     

    


    




    
    

