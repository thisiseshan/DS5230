#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:43:26 2024

@author: eshan
"""

from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from collections import Counter
from scipy.spatial import distance

#%%
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
data = mnist.data
target = mnist.target

# Convert the labels from string to integers
labels = target.astype(int)
# Reshape the data to a (number_of_samples, 28, 28) array
images = np.reshape(data, (-1, 28, 28))
images_normalized = images/255

#%%
# Display the first image
plt.imshow(images_normalized[0], cmap='gray')
plt.title(f'Label: {labels[0]}')
plt.show()


#%%
# Split into train,test,valid

X_train,X_temp,y_train,y_temp = train_test_split(images_normalized, labels, test_size=0.2)

X_val, X_test,y_val,y_test = train_test_split(X_temp, y_temp, test_size=0.5)
#%%
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

X_train_flattened.shape

#%%
# KNN library mode
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train_flattened, y_train)

accuracy = knn.score(X_test_flattened, y_test)

print(f"Accuracy : {accuracy * 100:.2f}%")


#%%
# KNN sexy mode

class sexy_knn:
    def __init__(self, k:int):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y.pred = [self._predict(x) for x in X]
    
    def _predict(self, x):
        distances = [distance.euclidean(x,x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
#%%
# Create KNN classifier
knn = sexy_knn(k=3)

# Fit the classifier to the data
knn.fit(X_train_flattended, y_train)

# Print the accuracy on the test and validation sets
print(f"Test Accuracy: {knn.score(X_test, y_test)}")
print(f"Validation Accuracy: {knn.score(X_val, y_val)}")
        
    # find distances
    # sort them
    # take top 3
    




    