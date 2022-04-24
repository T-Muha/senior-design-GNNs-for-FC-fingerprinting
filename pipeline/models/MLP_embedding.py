import torch
from torch import nn

class MLP_Embedding(nn.Module):
    """
    A MLP that independently embeds a pair of graphs, where
    similar graphs should be closer together in the embedding space.
    """

    def __init__(self):
        super(MLP_Embedding, self).__init__()
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(5050, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, adj_matrices):
        """Takes a pair of graph adjacency matrices and independently encodes them."""

        adj_flat = self.unpack_adj(adj_matrices)
        graph_rep = self.mlp(adj_flat)
        return graph_rep

    def unpack_adj(self, adj):
        """Returns flattened upper triangle of the adjacency matrices."""

        indices = torch.triu_indices(100,100)
        ind_x = indices[0]
        ind_y = indices[1]
        return adj[...,ind_x,ind_y]