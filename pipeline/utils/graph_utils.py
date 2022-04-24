from platform import node
import torch
import numpy as np
from numpy.random import uniform, randint

def knn_graph(graphs, k):
    """
    Return the knn graph of the given set of graphs.
    Still works if input graphs are batched/paired in some way.
    k is the number of strongest connections to keep.
    Note that these graphs are based on correlations, not distances.
    """

    num_nodes = graphs.shape[-1]
    mean_G = np.mean(graphs.reshape((-1,num_nodes,num_nodes)), axis=0)
    flat_G = np.triu(mean_G).flatten()

    least_indices = np.argsort(flat_G)[:-k]
    flat_G[least_indices] = 0

    knn_G = flat_G.reshape((num_nodes, num_nodes))
    knn_G = knn_G + knn_G.T
    binary_knn = np.where(knn_G > 0.001, 1, 0)

    return binary_knn


def random_walk(G, x, max_len):
  """
  Apply a random walk to graph G starting at node x.
  Returns a vector containing the path of the walk.
  """

  walk = [x]
  node = x
  for _ in range(max_len):
      neighbors = np.where(G[node] > 0)[0]
      if neighbors.shape[0]:
        node = neighbors[np.random.randint(0, neighbors.shape[0])]
        walk.append(node)
  return np.array(walk)


def random_walks(G, num_walks, max_len, window_size):
  """
  Conduct num_walks random walks on every node and use a sliding window
  to count the co-occurance of every node in the walks as a frequency matrix.
  """

  F = np.zeros_like(G)

  for _ in range(num_walks):
    for x in range(G.shape[0]):

      # Node must have at least one neighbor
      if np.sum(G[x]):
        walk = [x]
        node = x
        for _ in range(max_len):
          neighbors = np.where(G[node] > 0)[0]
          if len(neighbors) == 0:
            print("Something very wrong")
          index = np.random.randint(0, len(neighbors))
          node = neighbors[index]
          walk.append(node)

        # Update F with frequencies of nodes found in sliding window
        for i in range(len(walk)):
          co_nodes = walk[i:i+window_size]
          for node_1 in co_nodes:
            for node_2 in co_nodes:
              if node_1 != node_2:
                F[node_1, node_2] += 1
                F[node_2, node_1] += 1

  return F


def rand_walk_freq_matrix(G, num_walks, max_len, window_size):
  """
  Obtain frequency matrix of co-occurance of nodes within a window
  of n random walks.
  """

  F = np.zeros_like(G)

  for _ in range(num_walks):

    # Iterate all nodes as in paper, but this may be overkill
    for x in range(G.shape[0]):

      # Node must have at least one neighbor
      if np.sum(G[x]):
        walk = [x]
        node = x
        for _ in range(max_len):
          neighbors = np.where(G[node] > 0)
          if neighbors.shape[0] == 0:
            print("Something very wrong")
          node = neighbors[np.random.randint(0, neighbors.shape[0])]
          walk.append(node)

        # Update F with frequencies of nodes found in sliding window
        for i in range(len(walk)):
          co_nodes = walk[i:i+window_size]
          for node_1 in co_nodes:
            for node_2 in co_nodes:
              if node_1 != node_2:
                F[node_1, node_2] += 1
                F[node_2, node_1] += 1

  return F


def get_features(adj_matrices, node_dim, edge_dim):
    """
    Return the node features, edge features, and edge vertices for the
    given batches of pairs of adjacency matrices.
    """
    
    num_nodes = adj_matrices.shape[-1]
    batch_size = adj_matrices.shape[0]

    # Get node features
    # Vector of node_dim 1's for every node in every pair in every batch
    # Network unpacks when associating nodes with edges
    node_features = torch.ones(batch_size, 2, num_nodes, node_dim)

    indices = torch.tril_indices(100,100)
    ind_x = indices[0]
    ind_y = indices[1]
    adj_upper = adj_matrices.clone()
    adj_upper[...,ind_x,ind_y] = 0
  
    # Get edge features and vertices
    edge_vertices = torch.nonzero(torch.flatten(adj_upper, end_dim=-2)).long()

    edge_vertices[:,1] += edge_vertices[:,0] - edge_vertices[:,0] % 100
    edge_features = adj_upper[torch.nonzero(adj_upper, as_tuple=True)].repeat(edge_dim, 1).T

    return node_features, edge_features, edge_vertices