#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 11:52:37 2017

@author: thakkar_
"""
# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('churn.csv')
df.drop(labels = 'Unnamed: 0',axis = 1, inplace = True)
df.drop(labels = 'Phone',axis = 1, inplace = True)
df_NaN = df.replace('?',np.nan,inplace = True)
def boolToInt(a):
    if a == 'True.':
        return 1
    elif a == 'False.':
        return 0
df['Churn?'] = df['Churn?'].apply(lambda x: boolToInt(x))


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
df['State'] = labelencoder_X_1.fit_transform(df['State'])
labelencoder_X_2 = LabelEncoder()
df['VMail Plan'] = labelencoder_X_2.fit_transform(df['VMail Plan'])


X = df.drop(labels = 'Churn?', axis=1)
y = df.iloc[:, 20].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X['State'] = labelencoder_X_1.fit_transform(X['State'])
onehotencoder = OneHotEncoder(categorical_features = [1])
df = onehotencoder.fit_transform(df)
X = X[:, 1:]

# Visualizing

print(df.isnull().sum())
plt.figure(figsize=(13,10))
sns.heatmap(data = df.corr(), cmap= 'viridis')

# Splitting into TRAIN, TEST set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)