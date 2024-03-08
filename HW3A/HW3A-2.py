#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 18:06:45 2024

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
images_stanadardized = (images_normalized - np.mean(images_normalized, axis=0)) / np.std(images_normalized)
#%%
X_train, X_test, y_train, y_test = train_test_split(images_stanadardized.reshape((70000, 28*28)), labels, test_size=0.2, random_state=69)
X_train.shape
#%%
# PCA time
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
log_reg = LogisticRegression(penalty = 'l2', solver='lbfgs', max_iter=100, random_state=69)
decision_tree = DecisionTreeClassifier(random_state=69)
#%%
# MNIST PCA=5
pca_5 = PCA(n_components=5)
X_train_pca_5 = pca_5.fit_transform(X_train)
X_test_pca_5 = pca_5.transform(X_test)

log_reg.fit(X_train_pca_5, y_train)
decision_tree.fit(X_train_pca_5, y_train)

accuracy_log_reg_5 = accuracy_score(y_test, log_reg.predict(X_test_pca_5))
accuracy_dt_5 = accuracy_score(y_test, decision_tree.predict(X_test_pca_5))

# Evaluate models
accuracy_log_reg_5 = accuracy_score(y_test, log_reg.predict(X_test_pca_5))
accuracy_dt_5 = accuracy_score(y_test, decision_tree.predict(X_test_pca_5))

print("Accuracy Logisitc D=5", accuracy_log_reg_5)
print("Accuracy Decision Tree D=5", accuracy_dt_5)

#%%
# MNIST PCA=20
pca_20 = PCA(n_components=20)
X_train_pca_20 = pca_20.fit_transform(X_train)
X_test_pca_20 = pca_20.transform(X_test)

log_reg.fit(X_train_pca_20, y_train)
decision_tree.fit(X_train_pca_20, y_train)

accuracy_log_reg_20 = accuracy_score(y_test, log_reg.predict(X_test_pca_20))
accuracy_dt_20 = accuracy_score(y_test, decision_tree.predict(X_test_pca_20))

# Evaluate models
accuracy_log_reg_20= accuracy_score(y_test, log_reg.predict(X_test_pca_20))
accuracy_dt_20 = accuracy_score(y_test, decision_tree.predict(X_test_pca_20))

print("Accuracy Logisitc D=20", accuracy_log_reg_20)
print("Accuracy Decision Tree D=20", accuracy_dt_20)
#%%

# Load the spambase dataset

df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3A/dataset/spambase/spambase.csv')
df.shape
#%%
df_xtrain = df.iloc[0:4000, 0:57]
df_ytrain = df.iloc[0:4000, 57:58]

df_xtest = df.iloc[4000:4600,0:57]
df_ytest = df.iloc[4000:4600, 57:58]
#%%
# Do PCA
# MNIST PCA=5
pca_5 = PCA(n_components=5)
X_train_pca_5 = pca_5.fit_transform(df_xtrain)
X_test_pca_5 = pca_5.transform(df_xtest)

log_reg.fit(X_train_pca_5, df_ytrain)
decision_tree.fit(X_train_pca_5, df_ytrain)

accuracy_log_reg_5 = accuracy_score(df_ytest, log_reg.predict(X_test_pca_5))
accuracy_dt_5 = accuracy_score(df_ytest, decision_tree.predict(X_test_pca_5))

# Evaluate models
accuracy_log_reg_5 = accuracy_score(df_ytest, log_reg.predict(X_test_pca_5))
accuracy_dt_5 = accuracy_score(df_ytest, decision_tree.predict(X_test_pca_5))

print("Accuracy Logisitc D=5", accuracy_log_reg_5)
print("Accuracy Decision Tree D=5", accuracy_dt_5)

#%%
# Do PCA
# MNIST PCA=5
pca_20 = PCA(n_components=22)
X_train_pca_20 = pca_20.fit_transform(df_xtrain)
X_test_pca_20 = pca_20.transform(df_xtest)

log_reg.fit(X_train_pca_20, df_ytrain)
decision_tree.fit(X_train_pca_20, df_ytrain)

accuracy_log_reg_20 = accuracy_score(df_ytest, log_reg.predict(X_test_pca_20))
accuracy_dt_20 = accuracy_score(df_ytest, decision_tree.predict(X_test_pca_20))

# Evaluate models
accuracy_log_reg_20 = accuracy_score(df_ytest, log_reg.predict(X_test_pca_20))
accuracy_dt_20 = accuracy_score(df_ytest, decision_tree.predict(X_test_pca_20))

print("Accuracy Logisitc D=5", accuracy_log_reg_20)
print("Accuracy Decision Tree D=5", accuracy_dt_20)

#%%


