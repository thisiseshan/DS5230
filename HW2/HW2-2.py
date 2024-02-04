#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:09:18 2024

@author: eshan
"""

from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import confusion_matrix

#%%
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
data = mnist.data
target = mnist.target


labels = target.astype(int)

#%%

images = np.reshape(data, (-1, 28, 28))
#%%
images_normalized = images / 255.0
#%%
images_normalized[0]
#%%
images_flattened = images_normalized.reshape((70000, 28*28))
images_flattened = images_flattened.astype(np.float64)
#%%
images_flattened[0]
#%%
#%%


def Kmeans(K=10, data=images_flattened):
    nearest_centroids_indices = 0
    
    # pick random centroids
    initial_indices = np.random.choice(data.shape[0], K, replace=False)
    initial_centroids = data[initial_indices]
    initial_centroids = initial_centroids.astype(np.float64)
    new_centroids = np.zeros_like(initial_centroids)

    threshold = 1e-5
    
    
    while True:

        # E step

        distances = np.sqrt(((data[:, np.newaxis] - initial_centroids) ** 2).sum(axis=2))
        nearest_centroids_indices = np.argmin(distances, axis=1)
        
        # M Step
        for k in range(K):
            points_in_cluster = data[nearest_centroids_indices == k]
            
            if points_in_cluster.shape[0] > 0:
                new_centroids[k] = points_in_cluster.mean(axis=0)
            else:
                new_centroids[k] = data[np.random.choice(data.shape[0])]
        
        # handle that nan baby
        diff = np.linalg.norm(initial_centroids - new_centroids)
        if diff < threshold:
            break
        
        initial_centroids = new_centroids.copy()
    
    return new_centroids, nearest_centroids_indices

#%%
new_centroids, nearest_centroids_indices = Kmeans(10, images_flattened)
#%%
def plot_centroids(centroids, K=10):
    fig, axs = plt.subplots(1, K, figsize=(15,2))
    for i in range(K):
        ax = axs[i]
        centroid_image = centroids[i].reshape(28,28)
        ax.imshow(centroid_image, cmap='gray')
        ax.axis('off')
    plt.show()

plot_centroids(new_centroids)
#%%
def calculate_kmeans_objective(data, nearest_centroids_indices, centroids):
    total_wcss = 0  
    for k in range(centroids.shape[0]):  
        cluster_data = data[nearest_centroids_indices == k]
        if cluster_data.size > 0:
            squared_distances = np.sum((cluster_data - centroids[k])**2, axis=1)
            total_wcss += np.sum(squared_distances)
    return total_wcss

def calculate_purity(labels, nearest_centroids_indices):
    matrix = confusion_matrix(labels, nearest_centroids_indices)
    majority_sum = np.sum(np.amax(matrix, axis=0))
    purity = majority_sum / np.sum(matrix)
    return purity

def calculate_gini_index(labels, nearest_centroids_indices):
    matrix = confusion_matrix(labels, nearest_centroids_indices)
    total_samples = np.sum(matrix)
    gini_sum = 0
    for j in range(matrix.shape[1]):  
        cluster_size = np.sum(matrix[:, j])
        if cluster_size == 0:
            continue
        score = 1
        for i in range(matrix.shape[0]): 
            p_ij = matrix[i, j] / cluster_size
            score -= p_ij ** 2
        gini_sum += (cluster_size / total_samples) * score
    return gini_sum
#%%
# Calculate the KMeans objective
new_centroids, nearest_centroids_indices = Kmeans(10, images_flattened)
kmeans_objective = calculate_kmeans_objective(data=images_flattened, nearest_centroids_indices=nearest_centroids_indices, centroids=new_centroids)
print("KMeans Objective:", kmeans_objective)

purity = calculate_purity(labels, nearest_centroids_indices)
gini_index = calculate_gini_index(labels, nearest_centroids_indices)

print("Purity:", purity)
print("Gini Index:", gini_index)
plot_centroids(new_centroids, 10)
#%% 
# K = 5
new_centroids, nearest_centroids_indices = Kmeans(5, images_flattened)
kmeans_objective = calculate_kmeans_objective(data=images_flattened, nearest_centroids_indices=nearest_centroids_indices, centroids=new_centroids)
print("KMeans Objective):", kmeans_objective)

purity = calculate_purity(labels, nearest_centroids_indices)
gini_index = calculate_gini_index(labels, nearest_centroids_indices)

print("Purity:", purity)
print("Gini Index:", gini_index)
plot_centroids(new_centroids, 5)
#%%
new_centroids, nearest_centroids_indices = Kmeans(20, images_flattened)
kmeans_objective = calculate_kmeans_objective(data=images_flattened, nearest_centroids_indices=nearest_centroids_indices, centroids=new_centroids)
print("KMeans Objective:", kmeans_objective)

purity = calculate_purity(labels, nearest_centroids_indices)
gini_index = calculate_gini_index(labels, nearest_centroids_indices)

print("Purity:", purity)
print("Gini Index:", gini_index)
plot_centroids(new_centroids, 20)
#%%
#%%

#%%
# Load FashionMNIST

fastion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
data = fastion_mnist.data
target = fastion_mnist.target


labels = target.astype(int)
#%%
data.shape
#%%

images = np.reshape(data, (-1, 28, 28))
#%%
plt.imshow(images_normalized[0], cmap='gray')
plt.title(f'Label: {labels[0]}')
plt.show()
#%%
images_normalized = images / 255.0
#%%
images_normalized[0]
#%%
images_flattened = images_normalized.reshape((70000, 28*28))
images_flattened = images_flattened.astype(np.float64)

#%%
# Calculate the KMeans objective
new_centroids, nearest_centroids_indices = Kmeans(10, images_flattened)
kmeans_objective = calculate_kmeans_objective(data=images_flattened, nearest_centroids_indices=nearest_centroids_indices, centroids=new_centroids)
print("KMeans Objective:", kmeans_objective)

purity = calculate_purity(labels, nearest_centroids_indices)
gini_index = calculate_gini_index(labels, nearest_centroids_indices)

print("Purity:", purity)
print("Gini Index:", gini_index)
plot_centroids(new_centroids, 10)
#%% 
# K = 5
new_centroids, nearest_centroids_indices = Kmeans(5, images_flattened)
kmeans_objective = calculate_kmeans_objective(data=images_flattened, nearest_centroids_indices=nearest_centroids_indices, centroids=new_centroids)
print("KMeans Objective):", kmeans_objective)

purity = calculate_purity(labels, nearest_centroids_indices)
gini_index = calculate_gini_index(labels, nearest_centroids_indices)

print("Purity:", purity)
print("Gini Index:", gini_index)
plot_centroids(new_centroids, 5)
#%%
# K = 20
new_centroids, nearest_centroids_indices = Kmeans(20, images_flattened)
kmeans_objective = calculate_kmeans_objective(data=images_flattened, nearest_centroids_indices=nearest_centroids_indices, centroids=new_centroids)
print("KMeans Objective:", kmeans_objective)

purity = calculate_purity(labels, nearest_centroids_indices)
gini_index = calculate_gini_index(labels, nearest_centroids_indices)

print("Purity:", purity)
print("Gini Index:", gini_index)
plot_centroids(new_centroids, 20)
#%%

# NG20
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch the 20 newsgroups dataset
ng20 = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# Fit and transform the dataset
X = vectorizer.fit_transform(ng20.data)    
X_dense = X.toarray()

#%%
X_dense_subset = X_dense[1:1001]
labels_subset = labels[1:1001]  # Align labels with the subset

# Apply KMeans
new_centroids, nearest_centroids_indices = Kmeans(20, X_dense_subset)

# Calculate the KMeans objective for the subset
kmeans_objective = calculate_kmeans_objective(data=X_dense_subset, nearest_centroids_indices=nearest_centroids_indices, centroids=new_centroids)
print("KMeans Objective:", kmeans_objective)

# Calculate purity and Gini index for the subset
purity = calculate_purity(labels_subset, nearest_centroids_indices)
gini_index = calculate_gini_index(labels_subset, nearest_centroids_indices)

print("Purity:", purity)
print("Gini Index:", gini_index)
#%%




































































































































