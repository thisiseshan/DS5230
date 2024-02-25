#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 21:46:43 2024

@author: eshan
"""


# GOOGLE COLAB BABY!

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
def PCA(dataset, D):
    # data_standardised = (dataset - np.mean(dataset, axis=0)) / np.std(dataset, axis=0)
    
    Sigma = np.cov(dataset, rowvar=False)
    eigen_val, eigen_vec = np.linalg.eigh(Sigma)
    
    # desc_eigen_vec = np.sort(eigen_vec)[::-1]
    
    idx = eigen_val.argsort()[::-1]
    sorted_eigen_val = eigen_val(idx)
    sorted_eigen_vec = eigen_vec[:, idx]
    
    D_eigen_vec = sorted_eigen_vec[:,:D]
    
    x_reduced = np.dot(dataset, D_eigen_vec)

    return x_reduced

#%%
PCA(X_train[1:100], D=5)
#%%
