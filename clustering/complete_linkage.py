# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:28:22 2016

@author: ERLING
"""

import numpy as np

def complete_linkage(x):
    """
    complete_linkage(x) performs a complete-linkage hierarchical clustering of the dataset x using Euclidean metric.
    
    Inputs:
        - x: Set of observations (x1; x2; ...; xn) where each observation is a d-dimensional vector
        
    Outputs:
        - z: Vectors with classes assigned to each observation in x at each step, where class is in the range [0, k-1]
    """

    # Dimension of observations
    n = x.shape[0]
    d = x.shape[1]
    
    # Constriction of distance matrix D
    D = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            D[i,j] = np.linalg.norm(x[i,:]-x[j,:])
    
    # Number of clusters
    k = n
    
    # Assigning each point to its own cluster
    clusters = np.zeros(k)
    for i in range(n):
        clusters[i] = i
    
    z = []
    z.append(clusters)    
    
    while k > 1:
        
        # Finding the first minimum non-zero distance between two clusters
        indices = np.argwhere(D>0)
        min_dist = np.array([np.inf])
        for i in range(len(indices)):
            dist = D[indices[i,0], indices[i,1]]             
            if dist < min_dist[0]:
                min_dist = np.array([dist, indices[i,0], indices[i,1]])
        
        D = np.delete(D, [min_dist[1], min_dist[2]], axis=0)
        D = np.delete(D, [min_dist[1], min_dist[2]], axis=1)
        
        print D
        
        k = 0
                    