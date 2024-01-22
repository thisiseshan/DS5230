#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:24:24 2024

@author: eshan
"""
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from collections import Counter
#%%

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
data = mnist.data
target = mnist.target


labels = target.astype(int)

#%%

images = np.reshape(data, (-1, 28, 28))
images_normalized = images / 255.0

#%%
plt.imshow(images_normalized[0], cmap='gray')
plt.title(f'Label: {labels[0]}')
plt.show()
#%%

print("Before Normalization:", images[2][10:15, 10:15])
print("After Normalization:", images_normalized[2][10:15, 10:15])  

#%%
df = pd.DataFrame(images_normalized[0])
df.to_csv('images_normalized0.csv')

#%%
# Checking out pdist with library!!
from scipy.spatial.distance import pdist

distances = []
for im in images_normalized:
    distances.append(pdist(im, 'euclidean'))

#%%
image_norm_subset = images_normalized[0:10]
image_norm_subset.shape

#%%
from scipy.spatial.distance import pdist, squareform

flattened_images = image_norm_subset.reshape(image_norm_subset.shape[0], -1)
cosine_distances = pdist(flattened_images, metric='euclidean')


cosine_distances_matrix = squareform(cosine_distances)
cosine_similarity_matrix = 1 - cosine_distances_matrix

#%%
distances = pd.DataFrame(cosine_distances_matrix)
similarity = pd.DataFrame(cosine_similarity_matrix)

distances.to_csv('distances.csv')
similarity.to_csv('similarity.csv')
# Returns a 10x10 matrix for indicating similarity between each image

#%%

def eulcid_smart_mode(X,Y):
    squared_diff = X**2 + Y**2 - 2*X*Y
    sum_squared_diff = np.sum(squared_diff)

    # Take the square root of the sum
    dist = np.sqrt(sum_squared_diff)
    return dist

# image_norm_subset.shape

#%%
epic_dist = np.zeros((10,10))


for image in range(0,image_norm_subset.shape[0]):
    for image2 in range(image+1,image_norm_subset.shape[0]):
        epic_dist[image][image2] = eulcid_smart_mode(image_norm_subset[image], image_norm_subset[image2])
        
    

pd.DataFrame(epic_dist).to_csv('epic_dist.csv')

#%%
# COSINE SIMILARITY
from scipy.spatial.distance import cdist

# Flattenning for vector babY!!!
# flattened_images = images_normalized.reshape(images_norm_subset.shape[0], -1)


cosine_distances = cdist(flattened_images, flattened_images, metric='cosine')
cosine_similarity = 1 - cosine_distances
#%%

#%%

def eulcid_smart_mode(X, Y):
    X = X.reshape(-1, 1, X.shape[-1])
    Y = Y.reshape(1, -1, Y.shape[-1])
    print(X.shape)
    print(Y.shape)


    squared_diff = (X**2).sum(2) + (Y**2).sum(2) - 2 * np.dot(X, Y.T)
    squared_diff[squared_diff < 0] = 0

    # Calculate the Euclidean distance
    dist = np.sqrt(squared_diff)
    return dist

#%%
def euclidean_distance_matrix(X, Y):
    X_flat = X.reshape(X.shape[0], -1)
    Y_flat = Y.reshape(Y.shape[0], -1)
    sum_square_X = np.sum(X_flat ** 2, axis=1).reshape(-1, 1)
    sum_square_Y = np.sum(Y_flat ** 2, axis=1)
    squared_diff = sum_square_X + sum_square_Y - 2 * np.dot(X_flat, Y_flat.T)
    squared_diff[squared_diff < 0] = 0
    return np.sqrt(squared_diff)

def calculate_distances_in_batches(data, batch_size, reference_data):
    num_samples = data.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate how many batches are needed
    distances = np.zeros((num_samples, reference_data.shape[0]))

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_distances = euclidean_distance_matrix(data[start_idx:end_idx], reference_data)
        distances[start_idx:end_idx, :] = batch_distances

    return distances


batch_size = 1000
distances = calculate_distances_in_batches(images_normalized, batch_size, images_normalized)
#%%

import pandas as pd
pd.DataFrame(distances).to_csv('epic_dist2.csv')

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
y = labels
num_samples = distances.shape[0]
indices = np.arange(num_samples)

train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=69)
test_indices, val_indices = train_test_split(temp_indices, test_size=0.5, random_state=69)

knn_precomputed = KNearestNeighborsPrecomputed(k=3)
knn_precomputed.fit(distances, y)

print(f"Test Accuracy: {knn_precomputed.score(test_indices, y[test_indices])}")
print(f"Validation Accuracy: {knn_precomputed.score(val_indices, y[val_indices])}")


#%%





















