#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:51:09 2024

@author: eshan
"""

from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
#%%
circles_df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3A/dataset/threecircles.csv', names=['x', 'y','class'])
spirals_df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3A/dataset/twospirals.csv', sep='\t', names=['x', 'y','class'])
#%%
sns.scatterplot(x=circles_df['x'].astype(float), y=circles_df['y'].astype(float), hue=circles_df['class'])
sns.scatterplot(x=spirals_df['x'].astype(float), y=spirals_df['y'].astype(float), hue=spirals_df['class'])
#%%
X = circles_df[['x', 'y']]
y = circles_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69 )
#%%
model = LogisticRegression()
model.fit(X,y)
#%%
accuracy_score(y_test, model.predict(X_test))
#%%
X = spirals_df[['x', 'y']]
y = spirals_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69 )
#%%
model = LogisticRegression()
model.fit(X,y)
#%%
accuracy_score(y_test, model.predict(X_test))
#%%
def gaussian_kernel(X, sigma=1):
    X = X.to_numpy()
    sqDist = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2*np.dot(X,X.T)
    return np.exp(-sqDist / (2*sigma**2))

def gauss_kernel_PCA(dataset, D):
    K = gaussian_kernel(dataset, 1)
    
    n = K.shape[0]
    U = np.ones((n,n)) / n
    
    Kn = K - U.dot(K) - K.dot(U) + (U.dot(K)).dot(U)
    
    eigenval, eigenvec = np.linalg.eigh(Kn)
    
    # Top D components
    idx = eigenval.argsort()[::-1]
    eigenvec = eigenvec[:, idx]
    X_pca = eigenvec[:, :D]
    
    positive_eigenval = X_pca[X_pca > 0]
    
    
    # Normalize eigenval for projection
    eigenval = np.sqrt(positive_eigenval)
    X_pca = X_pca / eigenval.reshape(1,-1)
    
    return X_pca
#%%
X = circles_df[['x', 'y']]
y = circles_df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69 )
#%%
# For circles
gauss_kernel_PCA(X_train, D=3)








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    






