#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:51:09 2024

@author: eshan
"""

"""
A) Run KMeans on MNIST data (or a sample of it)
B) Run PCA on same data
C) Plot data in 3D with PCA representation with t=3 top eigen values; use shapes to to indicate truth digit label 
(circle, triangle, "+", stars, etc) and colors to indicate cluster ID (red blue green etc).
D) Select other 3 at random eigen values from top 20; redo the plot several times.
"""
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

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
X_sample = X_train[0:1000]
#%%
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X_sample)