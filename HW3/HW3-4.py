#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:57:13 2024

@author: eshan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
#%%

moon_df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3/moons.csv')
moon_df.rename(columns={"Xmoons_X1": 'x', "Xmoons_X2": "y"}, inplace=True)
data_array = moon_df.values
#%%
data_subset = data_array[:100]
data_subset.shape
#%%

#%%

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

def calculate_distance_matrix(data):
    n = len(data)
    distance_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1,n):
            distance_matrix[i][j] = euclidean_distance(data[i], data[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    return distance_matrix

#%%

distance_matrix = calculate_distance_matrix(data_subset)
#%%

#%%
def merge_closest_clusters(clusters, distance_matrix):
    min_distance = np.inf
    min_i, min_j = -1, -1
    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix)):
            if distance_matrix[i, j] < min_distance:
                min_distance = distance_matrix[i, j]
                min_i, min_j = i, j
 
    if min_i > min_j:
        min_i, min_j = min_j, min_i
    new_cluster = clusters[min_i] + clusters[min_j]
    new_clusters = [clusters[x] for x in range(len(clusters)) if x != min_i and x != min_j]
    new_clusters.append(new_cluster)
    return new_clusters, min_i, min_j, min_distance

def update_distance_matrix(distance_matrix, i, j):
   
    reduced_matrix = np.delete(distance_matrix, j, axis=0)
    reduced_matrix = np.delete(reduced_matrix, j, axis=1)

    for k in range(len(reduced_matrix)):
        if k >= i:
            adjusted_k = k if k < j else k + 1
            reduced_matrix[i, k] = reduced_matrix[k, i] = min(distance_matrix[i, adjusted_k], distance_matrix[j, adjusted_k])
    return reduced_matrix

def hierarchical_clustering(data, num_clusters):
    distance_matrix = calculate_distance_matrix(data)
    clusters = [[i] for i in range(len(data))]
    
    while len(clusters) > num_clusters:
        clusters, min_i, min_j, _ = merge_closest_clusters(clusters, distance_matrix)
        distance_matrix = update_distance_matrix(distance_matrix, min_i, min_j)  # Update uses original indices
        
    # Update cluster labeling to reflect final clusters
    cluster_labels = np.zeros(len(data))
    for idx, cluster in enumerate(clusters):
        for point in cluster:
            cluster_labels[point] = idx
            
    return cluster_labels

#%%
def plot_clusters(data, cluster_labels):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        cluster_data = data[cluster_labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {label}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Hierarchical Clustering with {len(unique_labels)} Clusters')
    plt.legend()
    plt.show()

#%%
# Initialize cluster
clusters = []

n = len(data_subset)

for i in range(n):
    clusters.append([i])


#%%

#%%
cluster_labels = hierarchical_clustering(data_subset, 2)
plot_clusters(data_subset, cluster_labels)
#%%

cluster_labels = hierarchical_clustering(data_subset, 5)
plot_clusters(data_subset, cluster_labels)
#%%

cluster_labels = hierarchical_clustering(data_subset, 10)
plot_clusters(data_subset, cluster_labels)








