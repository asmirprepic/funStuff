import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

class GraphMLPipeline:
  def __init__(self,grap):
    """Initalize the ML pipeline with a NetworkX graph
    This graph could by the RelationalToGraph class. 
    """
    self.graph = graph
    self.embeddings =  None

  def compute_node2vec_embeddings(self,dimensions = 64, walk_length =30, num_walks = 200, p = 1,q=1):
    """  
      Compute node embeddings using Node2vec. 
      The embeddings capture graph structure, including community,centrality and connection. 
      Returns a dictionary mapping node to embedding vector.

    """
    node2vec= Node2Vec(self.graph,dimensions = dimensions, walk_length = walk_length, num-walks = num_walks, p=p, q=q, workers = 4)
    model = node2vec.fit(window = 10, min_count =1)
    self.embeddings = {str(node): model.wv.get_vector(str(node)) for node in self.graph.nodes()}
    return self.embeddings

  def get_embedding_matrix_and_nodes(self):
    """
    Return an embedding matrix
    """
    if self.embeddings is None:
      raise ValueError("Embeddings have not been computed")
    nodes = list(self.embeddings.keys())
    matrix = np.array([self.embeddings[node] for node in nodes])
    return matrix,nodes

  def classify_nodes(self,labels,test_size = 0.3, random_state = 42):
    """
    Node classification using logistic regression
    labels: Dictionary mapping node id to class label.

    Returns:
    -----------
    Tuple (accuracy, trained classifier)

    """
    matrix,nodes = self.get_embedding_matrix_and_nodes()

    X = matrix
    y = np.array([labels(int)] for node in nodes])
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size = test_size, random_state = random_state)
    clf = LogisticRegression(max_iter = 1000)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    return accuracy, clf

  def visualize_embeddings_tsne(self, perplexity=30, n_iter=1000):
      """
      Visualize the node embeddings in 2D using t-SNE.
      """
      matrix, nodes = self.get_embedding_matrix_and_nodes()
      tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
      tsne_results = tsne.fit_transform(matrix)
      plt.figure(figsize=(8, 6))
      plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
      for i, node in enumerate(nodes):
          plt.annotate(node, (tsne_results[i, 0], tsne_results[i, 1]))
      plt.title("t-SNE Visualization of Node Embeddings")
      plt.xlabel("Dimension 1")
      plt.ylabel("Dimension 2")
      plt.show()

  def compute_node_anomaly_scores(self):
        """
        Compute a simple anomaly score based on the absolute deviation of a node's degree from the network average.
        Returns a dictionary mapping node id to its anomaly score.
        """
        degrees = dict(self.graph.degree())
        avg_degree = np.mean(list(degrees.values()))
        anomaly_scores = {node: abs(degree - avg_degree) for node, degree in degrees.items()}
        return anomaly_scores

  
  def cluster_nodes(self,n_clusters=3):
    """
    Cluster nodes using K-means on the embedding matrix
    Returns a dictionary mapping node id to cluster label
    """

    matrix,nodes = self.get_embedding_matrix_and_nodes()
    kmeans = KMeans(n_clusters = n_clusters,random_state = 42)
    clusters = kmeans.fit_predict(matrix)
    cluster_mapping = {int(node): cluster for node, cluster in zip(nodes,clusters)}
    return cluster_mapping

