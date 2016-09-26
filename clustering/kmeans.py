import numpy as np

def kmeans(k, x):
    """
    kmeans(k, x) performs a k-means clustering of the dataset z using k randomly selected means
    
    Inputs:
        - k: Number of desired clusters
        - x: Set of observations (x1, x2, ..., xn) where each observation is a d-dimensional vector
        
    Outputs:
        - z: Vector with classes assigned to each observation in x, where class is in the range [1, k]
    """

    # Dimension of observations
    d = x.shape[0]
    n = x.shape[1]
    
    # Constructin k random seeds with dimension d within range of dataset
    seeds = np.zeros(shape=(d,k))
    np.random.seed(2)
    for j in range(k):        
        for i in range(d):    
            seeds[i,j] = np.random.uniform(np.min(x), np.max(x))
    
    # Construction of list to hold the class vector for each iteration
    z = []
    
    # Construction of list to hold the means for each iteration
    means = []
    means.append(seeds)
    
    # Iteration process
    # Calculating the distance from each point to the k means
    # and assigning the point the class of the closest mean  
    
    converged = False
    maxIt = 50
    it = 0
    epsilon = 0.001
    
    while not converged and it < maxIt:
        converged = True
        
        flags = []
        # Looping over all observations
        for j in range(n):
            
            # Looping over each centroid and calculates distance
            dist = np.zeros(k)-1
            
            for c in range(k):
                dist[c] = np.linalg.norm((x[:,j]-means[it][:,c]), ord=2)

            # Flag the observation based on the minimum distance to a seed
            flags.append(np.argmin(dist)+1)
        
        z.append(np.array(flags))
        
        # Update process
        # Updating each seed to move to the centroid of the points
        # assigned to its class
        #for c in range()
        centroids = np.zeros(shape=(d,k))
        for c in range(k):            
            indices = np.where(z[it]==(c+1))[0]
            size = np.size(indices)
            if not size == 0:
                updated_centroids = np.sum(x[:, indices], axis=1)/size
                centroids[:,c] = updated_centroids
            else:
                print 'Size = 0'
                centroids[:,c] = means[-1][:,c]
        
        means.append(centroids)
        
        # If centroids does change, the algorithm has not converged
        if len(means) < 2:
            converged = False
        elif not np.sum([means[-1], -means[-2]]) < epsilon:           
            converged = False
        it += 1
    
    return (z, means)