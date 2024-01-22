#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 16:56:35 2024

@author: eshan
"""

from sklearn.datasets import fetch_20newsgroups
import math
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import pandas as pd

#%%

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch the 20 newsgroups dataset
ng20 = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# Fit and transform the dataset
X = vectorizer.fit_transform(ng20.data)
#%%
from sklearn.metrics.pairwise import euclidean_distances

# Convert sparse matrix to dense
X_dense = X.toarray()
distances_ng20 = euclidean_distances(X_dense)

#%%


class KNearestNeighborsPrecomputed:
    def __init__(self, k=3):
        self.k = k

    def fit(self, distances, y):
        self.distances = distances 
        self.y_train = y

    def predict(self, indices):
        y_pred = [self._predict(index) for index in indices]
        return np.array(y_pred)

    def _predict(self, index):
        k_indices = np.argsort(self.distances[index])[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def score(self, indices, y):
        predictions = self.predict(indices)
        return np.mean(predictions == y)
#%%

indices_ng20 = np.arange(distances_ng20.shape[0])

train_indices_ng20, temp_indices_ng20 = train_test_split(indices_ng20, test_size=0.2, random_state=42)
test_indices_ng20, val_indices_ng20 = train_test_split(temp_indices_ng20, test_size=0.5, random_state=42)

#%%
knn_ng20 = KNearestNeighborsPrecomputed(k=3)
knn_ng20.fit(distances_ng20, ng20.target)

# Evaluate the classifier
print(f"NG20 Test Accuracy: {knn_ng20.score(test_indices_ng20, ng20.target[test_indices_ng20])}")
print(f"NG20 Validation Accuracy: {knn_ng20.score(val_indices_ng20, ng20.target[val_indices_ng20])}")
#%%

























