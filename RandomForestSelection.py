# -*- coding: utf-8 -*-
"""
Created on Sat May 29 01:43:10 2021

@author: SONG
"""

import quandl as ql
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn import metrics



from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(x):
    vif = pd.DataFrame()
    vif["variables"] = x.columns
    vif["VIF"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    return vif

df = pd.read_csv("df.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.index = df["Date"]
df = df.drop(["Date"], axis = 1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

X = df.drop(["ret"], axis = 1)
y = df["ret"] * 0.01
y = pd.DataFrame(y)
y.loc[y["ret"] >= 0] = 1
y.loc[y["ret"] < 0 ] = 0
y = pd.DataFrame(y, dtype = int)

selection = SelectFromModel(RandomForestClassifier(n_estimators=1000))
selection.fit(X, y)
selection.get_support()



selection_support = selection.get_support()
col = X.columns

concat = np.stack((col, selection_support), axis = 1)
print(concat)

# Selected Spearman bond, mwtrv, etrav, toutv
# n = 3, cptrv, etrav, toutv, trvou, mwntd, mwnus, blchs, hrate, rtrbl, naddu ntrat, ntran 
# n = 250, blchs, ntrat, mirev, cptra, mktcp, (avbls)

