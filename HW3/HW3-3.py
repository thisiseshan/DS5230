#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:40:01 2024

@author: eshan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
#%%

class sexy_DBSCAN2:
    
    def __init__(self, epsilon=7.5, minPts=3):
        self.epsilon = epsilon
        self.minPts = minPts
        self.cluster_labels = []
    
    def fit_predict(self, X):
        self.cluster_labels = self._dbscan(X)
        return self.cluster_labels
    
    def _find_neighbours(self, points, point_idx):
        neighbours = []
        point_values = points.iloc[point_idx].values.reshape(1, -1)
        distances = cdist(point_values, points.values)
        neighbours = np.where(distances <= self.epsilon)[1]
        return neighbours.tolist()
    
    
    def _expand_cluster(self, points, point_idx, neighbours, cluster_label):
        self.cluster_labels[point_idx] = cluster_label
        i = 0
        while i < len(neighbours):
            neighbour_idx = neighbours[i]
            if self.cluster_labels[neighbour_idx] == -1:
                self.cluster_labels[neighbour_idx] = cluster_label
            elif self.cluster_labels[neighbour_idx] == 0:
                self.cluster_labels[neighbour_idx] = cluster_label
                neighbour_neighbours = self._find_neighbours(points, neighbour_idx)
                if len(neighbour_neighbours) >= self.minPts:
                    neighbours = neighbours + neighbour_neighbours
            i+=1
        

    
    def _dbscan(self, points):
        cluster_label = 0
        self.cluster_labels = [0] * len(points)
        
        for point_idx in range(len(points)):
            if self.cluster_labels[point_idx]==0:
                neighbours = self._find_neighbours(points, point_idx)
                if len(neighbours) < self.minPts:
                    self.cluster_labels[point_idx] = -1
                else:
                    cluster_label +=1
                    self._expand_cluster(points, point_idx, neighbours, cluster_label)
        
        return self.cluster_labels
        
        
        
#%%

# NG-20

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


ng20 = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# Fit and transform the dataset
X = vectorizer.fit_transform(ng20.data)    

print(X.shape)
#%%
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

print("Shape of Reduced Matrix:", X_reduced.shape)

X_subset = X_reduced[:1000]
#%%
dbscan = sexy_DBSCAN2(0.04, 6)
cluster_labels = dbscan.fit_predict(pd.DataFrame(X_subset))
#%%
plt.figure(figsize=(10, 8))
plt.scatter(X_subset[:, 0], X_subset[:, 1], c=cluster_labels, cmap='viridis', marker='o')
plt.title('DBSCAN Clustering on 20 Newsgroups (PCA-reduced)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster ID')
plt.show()

#%%



# Household data

file_path = '/Users/eshan/Documents/DS 5230/HW3/household_power_consumption.txt'  # Update this to your file path

df = pd.read_csv(file_path, sep=';', low_memory=False, infer_datetime_format=True, 
                 parse_dates={'datetime':[0,1]}, index_col=['datetime'], na_values=['?'])

print(df.head())
df.dropna(inplace=True)
#%%
from sklearn.preprocessing import StandardScaler

X = df[['Global_active_power', 'Global_reactive_power']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_subset = X_pca[:1000]

dbscan = sexy_DBSCAN2(0.5, 5)
cluster_labels = dbscan.fit_predict(pd.DataFrame(X_subset))

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_subset[:, 0], X_subset[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title('DBSCAN Clustering on Household Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster ID')
plt.show()

#%%

# Fashion-MNIST

class sexy_DBSCAN3:
    
    def __init__(self, epsilon=7.5, minPts=3):
        self.epsilon = epsilon
        self.minPts = minPts
        self.cluster_labels = []
    
    def fit_predict(self, X):
        self.cluster_labels = self._dbscan(X)
        return self.cluster_labels
    
    def _find_neighbours(self, points, point_idx):
        point_values = points[point_idx].reshape(1, -1)
        distances = cdist(point_values, points)
        neighbours = np.where(distances <= self.epsilon)[1]
        return neighbours.tolist()
    
    
    def _expand_cluster(self, points, point_idx, neighbours, cluster_label):
        self.cluster_labels[point_idx] = cluster_label
        i = 0
        while i < len(neighbours):
            neighbour_idx = neighbours[i]
            if self.cluster_labels[neighbour_idx] == -1:
                self.cluster_labels[neighbour_idx] = cluster_label
            elif self.cluster_labels[neighbour_idx] == 0:
                self.cluster_labels[neighbour_idx] = cluster_label
                neighbour_neighbours = self._find_neighbours(points, neighbour_idx)
                if len(neighbour_neighbours) >= self.minPts:
                    neighbours = neighbours + neighbour_neighbours
            i+=1

    
    def _dbscan(self, points):
        cluster_label = 0
        self.cluster_labels = [0] * len(points)
        
        for point_idx in range(len(points)):
            if self.cluster_labels[point_idx]==0:
                neighbours = self._find_neighbours(points, point_idx)
                if len(neighbours) < self.minPts:
                    self.cluster_labels[point_idx] = -1
                else:
                    cluster_label +=1
                    self._expand_cluster(points, point_idx, neighbours, cluster_label)
        
        return self.cluster_labels
        
    
#%%
from sklearn.datasets import fetch_openml

fastion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
data = fastion_mnist.data
target = fastion_mnist.target


labels = target.astype(int)
#%%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data)
#%%
dbscan = sexy_DBSCAN3(100, 4)
cluster_labels = dbscan.fit_predict(X_pca)
#%%
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
plt.title('DBSCAN Clustering on Household Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Cluster ID')
plt.show()

#%%
for cluster in np.unique(cluster_labels):
    if cluster == -1: continue
    

    indices = np.where(cluster_labels == cluster_labels)[0]
    if len(indices) > 10:
        indices = indices[:10]
    
    plt.figure(figsize=(20, 2))
    for i, idx in enumerate(indices):
        plt.subplot(1, 10, i + 1)
        image = data[idx].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.title(f'Cluster {cluster}')
        plt.axis('off')
    plt.show()
#%%










