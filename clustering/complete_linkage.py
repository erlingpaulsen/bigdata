# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 12:28:22 2016

@author: ERLING
"""

import numpy as np

def cluster_distance(c1, c2):
    """
    cluster_distance(c1, c2) computes the maximum distance between observations in cluster c1 and c2
    
    Inputs:
        - c1: Array of observations in cluster 1
        - c2: Array of observations in cluster 2
        
    Outputs:
        - max_dist: Maximum distance between any observation in cluster 1 and any observation in cluster 2
    """    
    
    max_dist = 0    
    for x1 in c1:
        for x2 in c2:
            dist = np.linalg.norm(x1-x2)
            if dist > max_dist:
                max_dist = dist
    return max_dist

def proximity_matrix(clusters):
    k = len(clusters)
    D = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            if i == j:
                d = 0
            else:
                d = cluster_distance(clusters[i], clusters[j])
            D[i,j] = d
    return D
    
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

    clusters = []
    for i in range(n):
        clusters.append(np.array([x[i,:]]))
    #clusters = np.array(clusters)
    
    # Number of clusters
    k = len(clusters)
    
    # Construction of proximity matrix D
    D = proximity_matrix(clusters)
    
    z = []
    z.append(np.copy(clusters))    
    
    # Sequence number and clustering level
    m = 0
    L = [0]
    while k > 1:

        # Finding the first minimum non-zero distance between two clusters
        indices = np.argwhere(D>0)
        min_dist = np.inf
        for i in range(len(indices)):
            dist = D[indices[i,0], indices[i,1]]             
            if dist < min_dist:
                min_dist = dist
                c1_idx = indices[i,0]
                c2_idx = indices [i,1]
        

        # Incrementing sequence number by 1 and setting the level of the sequence to min_dist
        m += 1
        L.append(min_dist)
        
        c1 = clusters[c1_idx]
        c2 = clusters[c2_idx]

        # Merging the two most similar clusters
        new_cluster = np.concatenate((c1, c2), axis=0)
        clusters = [c for i,c in enumerate(clusters) if i not in [c1_idx, c2_idx]]
        clusters.append(new_cluster)
                
        z.append(np.copy(clusters))     
        k = len(clusters)
        
        # Updating proximity matrix
        D = proximity_matrix(clusters)

    return (z, L)
                    