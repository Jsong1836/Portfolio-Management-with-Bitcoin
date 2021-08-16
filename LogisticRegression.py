# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 06:43:54 2021

@author: SONG
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def data_split(train, test, train_size, window_size):
    from sklearn.model_selection import train_test_split
    import pandas as pd
    X = []
    y = []
    
    train_values = pd.DataFrame(train).values
    test_values = pd.DataFrame(test).values
    threshold = int(len(test_values) * train_size)
    recursive = len(test_values) - threshold
    
    for i in range(recursive):
        training = train_values[0: (threshold + (i * window_size))]
        testing = test_values[0 : (threshold + (i * window_size))]
        X.append(training)
        y.append(testing)
    return np.array(X), np.array(y)

def normalize(data):
    data_mean = data.mean(axis = 0)
    data_std = data.std(axis = 0)
    return (data - data_mean) / data_std

df = pd.read_csv("data_source.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.index = df["Date"]
df = df.drop(["Date"], axis = 1)

X = df.drop(["ret", "price"], axis = 1)
y = df["ret"] * 0.01
y = pd.DataFrame(y)
y.loc[y["ret"] >= 0] = 1
y.loc[y["ret"] < 0 ] = 0
y = pd.DataFrame(y, dtype = int)

# n = 250, blchs, ntrat, mirev, cptra, mktcp, (avbls)

X1 = df[["blchs", "ntrat", "mirev", "cptra", "mktcp", "avbls"]]
corr = pd.concat([X1, y], axis = 1)
X1 = normalize(X1)

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.3, shuffle = False)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.2, shuffle = False)

num_features, num_classes = X_train.shape[0], X_train.shape[1]

training_steps = 1000
batch_size = 256
display_step = 50

train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

W = tf.Variable(tf.ones([num_features, num_classes]), name = "weight")
b = tf.Variable(tf.ones([num_classes]), name = "bias")

def LogisticRegression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)

def CrossEntropy(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth = num_classes)
    y_pred = tf.clib_by_value(y_pred, 1e-9, 1.)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.optimizers.SGD(lr = 0.0001)

def run_optimizer(x, y):
    with tf.GradientTape() as Tape:
        pred = LogisticRegression(x)
        loss = CrossEntropy(pred, y)
    
    gradients = Tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimizer(batch_x, batch_y)

    if step % display_step ==0:
        pred = LogisticRegression(batch_x)
        loss = CrossEntropy(pred, batch_y)
        accuracy = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, accuracy))

