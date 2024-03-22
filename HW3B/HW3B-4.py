#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:52:59 2024

@author: eshan
"""


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
#%%

# Fetch the 20 newsgroups dataset
ng20 = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')

# Fit and transform the dataset
X = vectorizer.fit_transform(ng20.data)    
X_dense = X.toarray()
labels = ng20.target
#%%
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state = 69)
#%%
# Now for 20NG Logistic
log_reg = LogisticRegression(penalty = 'l1', solver='saga', max_iter=100, random_state=69)
log_reg.fit(X_train, y_train)
#%%
coefficients = np.abs(log_reg.coef_).mean(axis=0)
top_200_indices = np.argsort(np.abs(coefficients))[-200:]

X_train_selected = X_train[:, top_200_indices]
X_test_selected = X_test[:, top_200_indices]
#%%
# Logistic Regression Classification
LR = LogisticRegression(max_iter=1000, random_state=69)
LR.fit(X_train_selected, y_train)
y_pred = LR.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
#%%
accuracy
