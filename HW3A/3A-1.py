#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:07:26 2024

@author: eshan
"""
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#%%
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
data = mnist.data
target = mnist.target
labels = target.astype(int)

images = np.reshape(data, (-1, 28, 28))
images_normalized = images / 255.0
images_normalized.shape
#%%
X_train, X_test, y_train, y_test = train_test_split(images_normalized.reshape((70000, 28*28)), labels, test_size=0.2, random_state=69)
X_train.shape
#%%
# Now for MNIST Logistic
log_reg = LogisticRegression(penalty = 'l2', solver='lbfgs', max_iter=100, random_state=69)
log_reg.fit(X_train, y_train)

coefficients = np.abs(log_reg.coef_).mean(axis=0)
top_30_indices_lr = np.argsort(coefficients)[-30:]
print(coefficients)
print(top_30_indices_lr)
#%%
# Now for MNIST decision tree
decision_tree = DecisionTreeClassifier(random_state=69)
decision_tree.fit(X_train, y_train)

importances_dt = decision_tree.feature_importances_
top_30_indices_dt = np.argsort(importances_dt)[-30:]
print(importances_dt)
print(top_30_indices_dt)
#%%
# Load the spambase dataset

df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3A/dataset/spambase/spambase.csv')
df.shape
#%%
df_xtrain = df.iloc[:, 0:57]
df_ytrain = df.iloc[:, 57:58]
#%%

# Now for Spambase Logistic
log_reg = LogisticRegression(penalty = 'l2', solver='lbfgs', max_iter=100, random_state=69)
log_reg.fit(df_xtrain, df_ytrain)

coefficients = np.abs(log_reg.coef_).mean(axis=0)
top_30_indices_lr = np.argsort(coefficients)[-30:]
print(coefficients)
print(top_30_indices_lr)
#%%
# Now for Spambase decision tree
decision_tree = DecisionTreeClassifier(random_state=69)
decision_tree.fit(df_xtrain, df_ytrain)

importances_dt = decision_tree.feature_importances_
top_30_indices_dt = np.argsort(importances_dt)[-30:]
print(importances_dt)
print(top_30_indices_dt)
#%%
# Load the 20NG dataset

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch the 20 newsgroups dataset
ng20 = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# Fit and transform the dataset
X = vectorizer.fit_transform(ng20.data)    
X_dense = X.toarray()

X_dense_subset = X_dense[1:1001]
labels_subset = labels[1:1001]  # Align labels with the subset
#%%

# Now for 20NG Logistic
log_reg = LogisticRegression(penalty = 'l2', solver='lbfgs', max_iter=100, random_state=69)
log_reg.fit(X_dense_subset, labels_subset)

coefficients = np.abs(log_reg.coef_).mean(axis=0)
top_30_indices_lr = np.argsort(coefficients)[-30:]
print(coefficients)
print(top_30_indices_lr)
#%%
# Now for 20NG decision tree
decision_tree = DecisionTreeClassifier(random_state=69)
decision_tree.fit(X_dense_subset, labels_subset)

importances_dt = decision_tree.feature_importances_
top_30_indices_dt = np.argsort(importances_dt)[-30:]
print(importances_dt)
print(top_30_indices_dt)

#%%
















