# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:17:34 2016

@author: http://shahramabyari.com/2015/12/30/my-first-attempt-with-local-outlier-factorlof-identifying-density-based-local-outliers/
"""

import numpy as np
#import pandas as pd
from sklearn.neighbors import NearestNeighbors
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import matplotlib.patches as mPatch
#from matplotlib.legend_handler import HandlerLine2D

#knn function gets the dataset and calculates K-Nearest neighbors and distances
def knn(df,k):
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(df)
    distances, indices = nbrs.kneighbors(df)
    return distances, indices

#reachDist calculates the reach distance of each point to MinPts around it
def reachDist(df,MinPts,knnDist):
    nbrs = NearestNeighbors(n_neighbors=MinPts)
    nbrs.fit(df)
    distancesMinPts, indicesMinPts = nbrs.kneighbors(df)
    distancesMinPts[:,0] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,1] = np.amax(distancesMinPts,axis=1)
    distancesMinPts[:,2] = np.amax(distancesMinPts,axis=1)
    return distancesMinPts, indicesMinPts

#lrd calculates the Local Reachability Density
def lrd(MinPts,knnDistMinPts):
    return (MinPts/np.sum(knnDistMinPts,axis=1))

#Finally lof calculates lot outlier scores
def lof(Ird,MinPts,dsts):
    lof=[]
    for item in dsts:
       tempIrd = np.divide(Ird[item[1:]],Ird[item[0]])
       lof.append(tempIrd.sum()/MinPts)
    return lof

#We flag anything with outlier score greater than 1.2 as outlier#This is just for charting purposes
def returnFlag(lofScores, knnindices):
    if x['Score']>1.2:
       return 1
    else:
       return 0

np.random.seed(1337)
cs = 10
x1 = np.random.multivariate_normal([10,10], [[1,0],[0,1]], cs)
x2 = np.random.multivariate_normal([1,1], [[1.2,0],[0,1.3]], cs)
x3 = np.random.multivariate_normal([5,5], [[1.2,0],[0,1.3]], 1)
x4 = np.random.multivariate_normal([9,9], [[1,0],[0,1]], 1)
X12 = np.concatenate((x1,x2), axis=0)
k = 5
X = np.concatenate((X12, x3, x4), axis=0)

f1 = plt.figure(figsize=(12,8), dpi=80, facecolor='w', edgecolor='k')
ax1 = f1.add_subplot(111)
ax1.plot(*zip(*X12), marker='.', color='b', markersize=12, ls='')
ax1.plot(*zip(*x3), marker='.', color='r', markersize=14, ls='')
ax1.plot(*zip(*x4), marker='.', color='g', markersize=14, ls='')
ax1.grid(True)
     
knndist, knnindices = knn(X,5)
reachdist, reachindices = reachDist(X,5,knndist)
irdMatrix = lrd(5,reachdist)
lofScores = lof(irdMatrix,5,reachindices)