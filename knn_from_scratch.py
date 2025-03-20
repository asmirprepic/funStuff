import numpy as np

def initialize_centroid(X: np.ndarray,k:int) -> np.ndarray:
  index = np.random.choice(X.shape[0],k,replace = False)
  return X[index]

def distance(X: np.ndarray,centroids: np.ndarray)-> np.ndarray:
  # Holder for distance
  distances = np.zeros(X.shape[0],len(centroids)))

  
  for i, centroid in enumerate(centroids):
    distances[:,i] = np.linalg.norm(X-centroid,axis = 1)
  return distances

def assing_cluster(X: np.ndarray,centroids: np.ndarray) -> np.ndarray:
  distances = distance(X,centroids)
  np.argmin(distances,axis = 1)

def update_centroid(X: np.ndarray,labels: np.ndarray,k: int) -> np.ndarray :
  new_centroids = np.zeros((k,X.shape[1]))
  for i in range(k):
    cluster_points = X[labels == i]
    if len(cluster_points) > 0:
      new_centroids[i] = cluster_points.mean(axis = 0)
  return new_centroids

def kmeans(X: np.ndarray,k: int,max_iters: int = 1000,tolerance: float = 1e-6) -> Tuple[np.ndarray,np.ndarray]:

  centroids = initialize_centroid(X,k)

  for _ in range(max_iters):
    labels = assign_clusters(X,centroids)

    new_centroids = update_centroid(X,labels, k)

    if np.linalg.norm(new_centroids-centroids) < tolerance: 
      break
    centroids = new_centroids

return centroid, labels
  
