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

# Importing the Dataset
data = pd.read_csv('churn.csv')

# Replacing all the '?' with np.nan
data_NaN = data.replace('?',np.nan,inplace = True)

# Setting the dependent Variabke 'Churn?' to 0's and 1's from False and True
def boolToInt(a):
    if a == 'True.':
        return 1
    elif a == 'False.':
        return 0
data['Churn?'] = data['Churn?'].apply(lambda x: boolToInt(x))

#Removing 'Unnamed: 0' and 'Phone' columns
data.drop(labels = 'Unnamed: 0',axis = 1, inplace = True)
data.drop(labels = 'Phone',axis = 1, inplace = True)

# Count of NaN values in the Dataset
print(data.isnull().sum())

# Visulalizing the dataframe
plt.figure(figsize=(13,10))
sns.heatmap(data = data.corr(), cmap= 'viridis')

# Encoding categorical data of the Dataframe
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
data['State'] = labelencoder_X_1.fit_transform(data['State'])
labelencoder_X_2 = LabelEncoder()
data['VMail Plan'] = labelencoder_X_2.fit_transform(data['VMail Plan'])






# To be done...
X = df.drop(labels = 'Churn?', axis=1)
y = df.iloc[:, 20].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X['State'] = labelencoder_X_1.fit_transform(X['State'])
onehotencoder = OneHotEncoder(categorical_features = [1])
df = onehotencoder.fit_transform(df)
X = X[:, 1:]




# Splitting into TRAIN, TEST set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)