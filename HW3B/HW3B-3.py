#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:29:09 2024

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

# X_dense_subset = X_dense[1:1001]
# labels_subset = labels[1:1001]
#%%
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state = 69)

chi2_selector= SelectKBest(chi2, k=200)
X_train_chi2 = chi2_selector.fit_transform(X_train, y_train)
X_test_chi2 = chi2_selector.transform(X_test)

# Decision Tree classificaiton
dt_classifier_chi2  = DecisionTreeClassifier(random_state=69)
dt_classifier_chi2.fit(X_train_chi2, y_train)
y_pred_dt_chi2 = dt_classifier_chi2.predict(X_test_chi2)
accuracy_dt_chi2 = accuracy_score(y_test, y_pred_dt_chi2)

# Logistic Regression Classification
lr_classifier_chi2 = LogisticRegression(max_iter=1000, random_state=69)
lr_classifier_chi2.fit(X_train_chi2, y_train)
y_pred_lr_chi2 = lr_classifier_chi2.predict(X_test_chi2)
accuracy_lr_chi2 = accuracy_score(y_test, y_pred_lr_chi2)


pca = PCA(n_components=200)
X_train_pca = pca.fit_transform(X_train.toarray())
X_test_pca = pca.transform(X_test.toarray())

# Classificaiton after PCA

# Decision Tree classificaiton
dt_classifier_chi2  = DecisionTreeClassifier(random_state=69)
dt_classifier_chi2.fit(X_train_pca, y_train)
y_pred_dt_pca = lr_classifier_chi2.predict(X_test_pca)
accuracy_dt_pca = accuracy_score(y_test, y_pred_dt_pca)

# Logistic Regression Classification
lr_classification_chi2 = LogisticRegression(max_iter=1000, random_state=69)
lr_classifier_chi2.fit(X_train_pca, y_train)
y_pred_lr_pca = lr_classifier_chi2.predict(X_test_chi2)
accuracy_lr_pca = accuracy_score(y_test, y_pred_lr_pca)
#%%
(accuracy_dt_chi2, accuracy_lr_chi2, accuracy_dt_pca, accuracy_lr_pca)


#%%
