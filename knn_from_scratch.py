import numpy as np

def initialize_centroid(X,k):
  index = np.random.choice(X.shape[0],k,replace = False)
  return X[index]

def distance(X,centroids):
  
