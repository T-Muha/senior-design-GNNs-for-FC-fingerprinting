from tracemalloc import start
import torch
from torch import nn

class MLP_Diff_Similarity(nn.Module):
    """
    MLP predicting the similarity of two input graphs.
    Maps a the difference of a pair of graphs to a similarity value.
    """

    def __init__(self):
        super(MLP_Diff_Similarity, self).__init__()

        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(5050, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, adj_matrices):
        """Compute similarity of the input graphs."""

        adj_flat = self.unpack_adj(adj_matrices)
        similarity = self.mlp(adj_flat)
        return similarity

    def unpack_adj(self, adj):
        """
        Returns concatenation of flattened upper triangles of
        adjacency matrices in a pair.
        """

        indices = torch.triu_indices(100,100)
        ind_x = indices[0]
        ind_y = indices[1]
        adj_flat = torch.flatten(adj[...,ind_x,ind_y], start_dim=-1)

        return adj_flat