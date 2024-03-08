#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:22:40 2024

@author: eshan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import seaborn as sns
from sklearn.decomposition import PCA
#%%
# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
data = mnist.data
target = mnist.target
#%%

labels = target.astype(int)
images = np.reshape(data, (-1, 28, 28))
images_normalized = images / 255.0
images_normalized[0]
images_flattened = images_normalized.reshape((70000, 28*28))
images_flattened = images_flattened.astype(np.float64)
#%%
images_subset = images_flattened[0:1000]
labels_subset = labels[0:1000]
#%%
images_flattened.shape
#%%
pca = PCA(n_components=50)
X_pca = pca.fit_transform(images_subset)
#%%
n,d = X_pca.shape
#%%
# Define variables
perplexity = 30

def Hbeta(D=np.array([]), beta = 1):
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H,P


#%%
# compute pairwaise

sum_X_pca = np.sum(np.square(X_pca), axis=1)

# Compute squared Euclidean distance matrix
D = -2 * np.dot(X_pca, X_pca.T) + sum_X_pca + sum_X_pca[:, np.newaxis]

# now probabilites from distance amtrix
P = np.zeros((n,n))
beta = np.ones((n,1))
beta = np.ones((n, 1))
logU=np.log(perplexity)
tol = 1e-5

for i in range(n):

    betamin = -np.inf
    betamax = np.inf
    Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
    (H, thisP) = Hbeta(Di, beta[i])
    
    # Check if perplexity is within tolerance
    Hdiff = H-logU
    tries = 0
    while np.abs(Hdiff) > tol and tries < 50:
        if Hdiff > 0:
            betamin = beta[i].copy()
            if betamax == np.inf or betamax == -np.inf:
                beta[i] = beta[i] * 2
            else:
                beta[i] = (beta[i] + betamax) / 2
        else:
            betamax = beta[i].copy()
            if betamin == np.inf or betamin == -np.inf:
                beta[i] = beta[i] / 2
            else:
                beta[i] = (beta[i] + betamin) / 2
        
        (H,thisP) = Hbeta(Di, beta[i])
        Hdiff = H - logU
        tries += 1

    P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    
    print("Mean value of sigma %f" % np.mean(np.sqrt(1/beta)))
    
# now we have P

P = P + np.transpose(P)
P = P / np.sum(P)
P = P * 4
P = np.maximum(P, 1e-12)

min_gain = 0.01
initial_momentum = 0.5
final_momentum = 0.8
no_dims = 2
Y = np.random.randn(n, no_dims) * 1e-4
dY = np.zeros((n, no_dims))
iY = np.zeros((n, no_dims))
gains = np.ones_like(Y)
eta = 500


for iter in range(1000):
    
    # Computing pairwaise affinities
    
    sum_Y = np.sum(np.square(Y), axis=1)
    num = -2 * np.dot(Y, Y.T)
    num = 1 / (1 + np.add(np.add(num, sum_Y).T, sum_Y))
    num[range(n), range(n)] = 0
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    
    # Gradient
    
    PQ = P - Q
    for i in range(n):
        dY[i, :] = 4 * np.sum(np.tile(PQ[:, i] , (no_dims, 1)).T * (Y[i,:] - Y), axis=0)
        
    if iter < 20:
        momentum = initial_momentum
    else:
       momentum = final_momentum
       
    gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0)==(iY > 0))
    gains[gains < min_gain] = min_gain
    iY = momentum * iY - eta * (gains * dY)
       
    Y = Y + iY
    Y = Y - np.tile(np.mean(Y, 0), (n, 1))      
 
    if (iter + 1) % 100 == 0:
        C = np.sum(P * np.log(P/Q))
        print("Iteration %d error is %f" % (iter + 1, C))
        

    if iter == 100:
        P = P/4
        
print(Y)
#%%
Y
#%%

plt.figure(figsize=(10,8))
sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=labels_subset, palette=sns.color_palette("hsv", 10), legend='full')
plt.title('MNIST TSNe')
plt.show()


#%%
