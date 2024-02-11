#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:53:43 2024

@author: eshan
"""

# Naive
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import pandas as pd
import numpy as np
from scipy.stats import binom
#%%
data_list = []
with open('/Users/eshan/Documents/DS 5230/HW2/2gaussian.txt') as file:
    for line in file:
        parts = line.split()
        data_list.append([float(part) for part in parts])
        
data_array = np.array(data_list)
#%%
data_array
#%%
n_components = 2

gmm = GaussianMixture(n_components=n_components, random_state=42)

# Assuming X is your data, fit the model
gmm.fit(data_array)
#%%

labels = gmm.predict(data_array)
#%%

# After fitting, the model has estimated the mean and covariance
estimated_mean = gmm.means_[0]
estimated_cov = gmm.covariances_[0]
#%%
estimated_mean, estimated_cov
#%%
gmm.means_[1]
#%%
probabilities = gmm.predict_proba(data_array)
# Get the probabilities (responsibilities) of each point for the single component
probabilities = gmm.predict_proba(data_array)

# For a single-component GMM, this gives a probability of 1 for the component for all points
# since probabilities[:, 0] will be an array of 1s
print("Probabilities of points belonging to the component:", probabilities[:, 0])

# Threshold for belonging could be set, e.g., 0.9 for multi-component models
threshold = 0.9
count_above_threshold = (probabilities[:, 0] > threshold).sum()
print(f"Number of points with probability > {threshold}:", count_above_threshold)


# Count how many points are assigned to the single component
# This will be equal to the total number of points in a single-component GMM
print("Total number of points:", len(data_array))

#%%
def e_step(X, pi, mu, sigma, K):
    # mu given minimize distance
    N, K = X.shape[0], len(pi)
    gamma = np.zeros((N,K))

    for k in range(K):
        gamma[:, k] = pi[k] * sexy_multivariate_normal(X, mu=mu[k], cov=sigma[k], K=K)

    gamma /= np.sum(gamma, axis = 1, keepdims=True)
    return gamma

#%%
def m_step(X, gamma):
    # pi given average out and find mu
    N,D = X.shape
    K = gamma.shape[1]
    mu = np.zeros((K,D))
    sigma = np.zeros((K,D,D))
    pi = np.zeros(K)

    for k in range(K):
        Nk = np.sum(gamma[:, k])
        mu[k] = np.sum(gamma[:, k, np.newaxis] * X, axis=0) / Nk
        sigma[k] = (gamma[:, k, np.newaxis] * (X - mu[k])).T @ (X - mu[k]) / Nk
        pi[k] = Nk / N

    return pi, mu, sigma

#%%
def sexy_multivariate_normal(X, mu, cov, K):
    return (1 / (np.sqrt(2 * 3.14)**K*np.linalg.det(cov))) * (np.exp(np.dot(np.dot((-0.5)*(X-mu).T,np.linalg.inv(cov)),(X-mu))))
#%%
def compute_log_likelihood(X, pi, mu, sigma, W):
    N,K = X.shape[0], len(pi)
    log_likelihood = 0

    for n in range(N):
        temp = 0
        for k in range(K):
            temp += pi[k] * sexy_multivariate_normal(X[n], mu[k], sigma[k], W)
        log_likelihood += np.log(temp)

    return log_likelihood

#%%
def GMM(X, K, max_iters=100, tol=1e-4):
    N,D = X.shape
    mu = np.random.rand(K,D)
    sigma = np.array([np.eye(D) for _ in range(K)])
    pi = np.ones(K) / K

    log_likelihood_old = -np.inf

    for i in range(max_iters):
        gamma = e_step(X,pi, mu, sigma, K)

        pi, mu, sigma = m_step(X, gamma)

        log_likelihood_new = compute_log_likelihood(X, pi,mu,sigma, K)

        if np.abs(log_likelihood_new - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood_new

    n_points_per_component = np.sum(gamma, axis=0)

    return mu, sigma, n_points_per_component
    # call e and m

#%%
mu, sigma, n_points = GMM(data_array, K=2)
#%%
print("Mean: ", mu)
print("Covariance: ", sigma)
print("Number of points: ", n_points)
#%%
data_list = []
with open('/Users/eshan/Documents/DS 5230/HW2/3gaussian.txt') as file:
    for line in file:
        parts = line.split()
        data_list.append([float(part) for part in parts])

data_array2 = np.array(data_list)
#%%
mu2, sigma2, n_points2 = GMM(data_array2, K=3)
#%%
print("Mean: ", mu2)
print("Covariance: ", sigma2)
print("Number of points: ", n_points2)
#%%













#%%
