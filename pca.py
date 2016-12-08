# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 10:35:14 2016

@author: ERLING
"""

import numpy as np
import pandas as pd
import datahandler as dh
import filehandler as fh

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

filename = 'FULL_PERIOD_ROUNDED_SELECTED_VARS_FILTERED_INTERP_SUM_1MIN.h5'

df = pd.read_hdf('data/FULL_PERIOD_ROUNDED_SELECTED_VARS_FILTERED_INTERP_SUM_1MIN.h5', 'table')

df = df.dropna(axis=0)

X = df.values

# Scaling data to have zero mean and unit variance
X_norm = scale(X, axis=0, with_mean=True, with_std=True)

# Removing first 529 rows where shaft thrust and torque is exactly zero
X_norm = X_norm[529:,:]

# Computing the first 6 principal components and the scores
pca = PCA(n_components=10).fit(X_norm)
scores = pca.transform(X_norm)


# Clustering our dataset in 2 clusters, using only every 5th sample (due to memory)
X_norm2 = []
for i,x in enumerate(X_norm):
    if i%5==0:
        X_norm2.append(x)
X_norm2 = np.array(X_norm2)
scores2 = pca.transform(X_norm2)

ward = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward', compute_full_tree=False).fit(X_norm2)
kmeans = KMeans(n_clusters=2).fit(X_norm)
