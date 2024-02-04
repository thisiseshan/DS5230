import numpy as np
from scipy.special import comb
from scipy.stats import binom
#%%

def e_step(X, P, Pi, D):
    # mu given minimize distance
    N = X.shape[0]
    K = len(P)
    gamma = np.zeros((N,K))

    for k in range(K):
        for n in range(N):
            x = np.sum(X[n])
            gamma[n, k] = Pi[k] * binom.pmf(x, D, P[k])

    gamma /= np.sum(gamma, axis = 1, keepdims=True)
    return gamma

#%%
def m_step(X, gamma):
    # pi given average out and find mu
    N,D = X.shape
    K = gamma.shape[1]
    P = np.zeros(K)
    Pi = np.zeros(K)

    for k in range(K):
        P[k] = np.sum(gamma[:, k] * np.sum(X, axis=1)) / (D*np.sum(gamma[:, k]))
        Pi[k] = np.sum(gamma[:, k]) / N

    return P, Pi

#%%
def compute_log_likelihood(X, P, Pi, D):
    N,K = X.shape[0], len(Pi)
    log_likelihood = 0

    for n in range(N):
        temp = 0
        x = np.sum(X[n])
        for k in range(K):
            temp += Pi[k] * binom.pmf(x, D, P[k])
        log_likelihood += np.log(temp)

    return log_likelihood

#%%
def EM_COIN(X, D, K, max_iters=100, tol=1e-4):

    P = np.random.rand(K)
    Pi = np.array([0.33,0.33,0.33])
    p_a, p_b, p_c = 0.5,0.5,0.5
    piA, piB, piC = 1/3,1/3,1/3


    log_likelihood_old = -np.inf

    for i in range(max_iters):
        gamma = e_step(X, P, Pi, D)

        P, Pi = m_step(X, gamma)

        log_likelihood_new = compute_log_likelihood(X, P, Pi, D)

        if np.abs(log_likelihood_new - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood_new

    # n_points_per_component = np.sum(gamma, axis=0)


    return P, Pi
    # call e and m

#%%
data = []
with open('/Users/eshan/Documents/DS 5230/HW2/data.txt') as file:
    for line in file:
        session = [int(flip) for flip in line.strip().split()]
        data.append(session)

# Optionally, convert the list of lists into a NumPy array for efficient processing
data_array = np.array(data)

#%%
P, Pi = EM_COIN(data_array, 20, 3)
#%%
print("Probability: ", P)
print("Pi aka Responsibility: ", Pi)
#%%
data_array.shape
#%%
