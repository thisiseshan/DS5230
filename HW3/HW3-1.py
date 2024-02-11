#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:38:32 2024

@author: eshan
"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
#%%
dbscan_df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3/dbscan.csv')

#%%
dbscan_df.head()
#%%
epsilon = 7.5
min_samples = 3
dbscan = DBSCAN(epsilon, min_samples = min_samples)
dbscan_df['cluster'] = dbscan.fit_predict(dbscan_df[['x', 'y']])
#%%

dbscan_df.head()

#%%

import matplotlib.pyplot as plt

# Visualization
plt.figure(figsize=(10, 6))

# Scatter plot of the data points, color-coded by cluster ID
# Points with cluster ID -1 are considered noise and marked in black
plt.scatter(dbscan_df['x'], dbscan_df['y'], c=dbscan_df['cluster'], cmap='viridis', marker='o', s=100, label='Data Points')
plt.colorbar(label='Cluster ID')

# Mark noise points specifically
noise_points = dbscan_df[dbscan_df['cluster'] == -1]
plt.scatter(noise_points['x'], noise_points['y'], color='red', marker='x', s=100, label='Noise')

plt.title('DBSCAN Clustering')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()

plt.show()
#%%

class sexy_DBSCAN:
    
    def __init__(self, epsilon=7.5, minPts=3):
        self.epsilon = epsilon
        self.minPts = minPts
        self.cluster_labels = []
    
    def fit_predict(self, df):
        points = df[['x', 'y']].values
        self.cluster_labels = self._dbscan(points)
        return self.cluster_labels
    
    def _find_neighbours(self, points, point_idx):
        neighbours = []
        distances = cdist(points[point_idx].reshape(1, -1), points)
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
        
    def plot(self, df):
        plt.figure(figsize=(10, 6))
        plt.scatter(df['x'], df['y'], c=self.cluster_labels, cmap='viridis', marker='o', s=100)
        plt.title('DBSCAN Clustering')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.colorbar(label='Cluster ID')
        plt.show()

#%%
dbscan = sexy_DBSCAN(7.5, 3)
#%%
dbscan.fit_predict(dbscan_df)
#%%
dbscan.plot(dbscan_df)
#%%

## Setup circles, blobs and moons

circles_df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3/circles.csv')
blobs_df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3/blobs.csv')
moon_df = pd.read_csv('/Users/eshan/Documents/DS 5230/HW3/moon.csv')

circles_df.rename(columns={"Xcircle_X1": 'x', "Xcircle_X2": "y"}, inplace=True)
blobs_df.rename(columns={"Xblobs_X1": 'x', "Xblobs_X2": "y"}, inplace=True)
moon_df.rename(columns={"Xmoons_X1": 'x', "Xmoons_X2": "y"}, inplace=True)
#%%
dbscan = sexy_DBSCAN(0.1, 4)
dbscan.fit_predict(circles_df)
dbscan.plot(circles_df)
#%%
dbscan = sexy_DBSCAN(0.5, 4)
dbscan.fit_predict(blobs_df)
dbscan.plot(blobs_df)
#%%
dbscan = sexy_DBSCAN(0.2, 4)
dbscan.fit_predict(moon_df)
dbscan.plot(moon_df)











