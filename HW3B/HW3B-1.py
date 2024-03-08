#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:36:30 2024

@author: eshan
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import seaborn as sns
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
tsne = TSNE(n_components=2, random_state=69)
X_tsne = tsne.fit_transform(images_flattened)
#%%
plt.figure(figsize=(10,8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette=sns.color_palette("hsv", 10), legend='full')
plt.title('MNIST TSNe')
plt.show()
#%%

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch the 20 newsgroups dataset
ng20 = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# Fit and transform the dataset
X = vectorizer.fit_transform(ng20.data)    
X_dense = X.toarray()

X_dense_subset = X_dense[1:1001]
labels_subset = labels[1:1001]
#%%

tsne = TSNE(n_components=2, random_state=69)
X_tsne2 = tsne.fit_transform(X_dense_subset)
#%%
plt.figure(figsize=(10,8))
sns.scatterplot(x=X_tsne2[:, 0], y=X_tsne2[:, 1], hue=labels_subset, palette=sns.color_palette("hsv", 10), legend='full')
plt.title('MNIST TSNe')
plt.show()










