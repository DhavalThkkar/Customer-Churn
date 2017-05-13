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

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc,roc_auc_score

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold

# Importing the Dataset
data = pd.read_csv('churn.csv')

#Removing 'Unnamed: 0' and 'Phone' columns
data.drop('Unnamed: 0',axis = 1, inplace = True)
data.drop('Phone',axis = 1, inplace = True)
data.drop('Area Code', axis = 1, inplace = True) # Area code doesn't appear to be useful

# Replace ? so label encoding works
data['VMail Plan'].replace('?','no',inplace=True)
data['Int\'l Plan'].replace('?','no',inplace=True)

le = LabelEncoder()
data['VMail Plan'] = le.fit_transform(data['VMail Plan'])
data['Int\'l Plan'] = le.fit_transform(data['Int\'l Plan'])

# Check for ?, replace with NaN (float value)
data.replace('?', 0, inplace=True)

# There are some large values which could be concerning
col_names = ['VMail Message', 'Account Length', 'Day Mins', 'Day Calls', 'Day Charge', 'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls', 'Night Charge', 'Intl Mins', 'Intl Calls', 'Intl Charge']
data[col_names] = data[col_names].astype(float)
for col in col_names:
    data[col] = (data[col] - data[col].mean()) / data[col].std()

# Create your train and test data sets
data['Churn?'] = data['Churn?'].apply(lambda x: x.split('.')[0])
y = le.fit_transform(data['Churn?'])
data.drop('Churn?', axis=1, inplace=True)

state = pd.get_dummies(data['State'])
data.drop('State', axis=1, inplace=True)
X = pd.concat([data, state], axis=1).as_matrix()

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)    
for train, test in kfold.split(X, y):
    
    """
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X[train] = sc.fit_transform(X[train])
    X[test] = sc.transform(X[test])"""
    # Initial tests appear to indicate no overfitting, dropout layer unneccesary
    ann = Sequential()    
    ann.add(Dense(100, activation='tanh', kernel_initializer='random_normal', input_shape=(X[train].shape[1],)))
    ann.add(Dropout(rate=0.1))
    ann.add(Dense(100, activation='tanh', kernel_initializer='random_normal'))
    ann.add(Dropout(rate=0.1))
    ann.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ann.fit(X[train], y[train], batch_size=100, epochs=150, verbose=0)
    fpr, tpr, _ = roc_curve(y[test],ann.predict_proba(X[test]))
    auc_score = auc(fpr,tpr)
    
    print('AUC Score: {:.3f}%'.format(auc_score))
    print('State Dummy Variables Accuracy: {:.2f}%'.format(ann.evaluate(X[test], y[test], verbose=0)[1]*100))

# Tuning the ANN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = units, kernel_initializer = 'random_normal', activation = 'tanh', input_shape = (X_train.shape[1],)))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = units, kernel_initializer = 'random_normal', activation = 'tanh'))
    classifier.add(Dropout(rate = 0.1))
    classifier.add(Dense(units = 1, kernel_initializer = 'random_normal', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [50,100,150,200,250,300],
              'units' : [50,100,150],
              'epochs': [100, 500],
              'optimizer': ['adam']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_