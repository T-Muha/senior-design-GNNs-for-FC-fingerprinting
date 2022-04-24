import torch
from torch import nn
import scipy

class SpectralFilterLayer(nn.Module):
    """
    Layer that applies spectral filtering to input feature maps as
    convolutions to output feature maps
    """

    def __init__(self, laplacian, k, filters_in, filters_out):
        super(SpectralFilterLayer, self).__init__()

        self.k = k
        self.laplacian = laplacian
        self.convolutions = nn.Sequential(
            nn.Linear(k * filters_in, filters_out),
            nn.ReLU()
        )
        return

    def forward(self, device, features):
        """Process input feature maps."""

        laplacian = self.laplacian.to(device)
        polynomials = [features, torch.matmul(laplacian, features)]

        # Compute Chebyshev polynomials
        for _ in range(2, self.k):
            T_k = 2 * torch.matmul(laplacian, polynomials[-1]) - polynomials[-2]
            polynomials.append(T_k)

        conv_input = torch.cat(polynomials, -1)
        features_out = self.convolutions(conv_input)

        return features_out


class S_GCN(nn.Module):
    """
    Model implementing the Siamese graph convolutional network described in "Deep Learning
    for Graph Similarity Analysis." A pair of Siamese GCNs each process one of the graphs in
    the input pair. Final GCN feature maps from each branch are combined in the dot product
    layer and mapped to a similarity value in the similarity layer.
    """

    def __init__(self, layer_sizes, knn_G):
        super(S_GCN, self).__init__()

        self.k = 3
        self.use_full = False
        self.num_nodes = knn_G.shape[-1]

        laplacian = torch.from_numpy(scipy.sparse.csgraph.laplacian(knn_G, normed=True)).type(torch.float32)
        eigenvalues, _ = torch.linalg.eigh(laplacian)
        scaled_laplacian = 2 * laplacian / eigenvalues[-1] - torch.eye(laplacian.shape[-1])

        self.filter_layers = nn.ModuleList([SpectralFilterLayer(scaled_laplacian, self.k, 1, layer_sizes[0])])
        for i in range(1, len(layer_sizes)):
            self.filter_layers.append(SpectralFilterLayer(scaled_laplacian, self.k, layer_sizes[i-1], layer_sizes[i]))

        # Take upper triangle of the symmetrical combined output and map to similarity value
        triu = int((self.num_nodes ** 2 + self.num_nodes) / 2) - self.num_nodes
        self.similarity_layer = nn.Linear(triu, 1)
        self.dropout = nn.Dropout(0.8)
        
    def forward(self, device, graph_pairs):
        """
        Apply spectral filters to each graph of pair in batch,
        combine graphs in pair with dot product layer, and compute 
        similarity score from similarity layer.
        """

        # Last dimension is feature map
        graph_pairs = graph_pairs.unsqueeze(-1)

        # Apply spectral filters
        filtered = self.filter_layers[0](device, graph_pairs)
        if len(self.filter_layers) > 1:
            for layer in self.filter_layers[1:]:
                filtered = layer(device, filtered)

        # Combine pairs and features through dot product
        combined = (filtered[:,0] * filtered[:,1]).sum(dim=-1)
        
        if self.use_full:
            combined = combined.flatten(start_dim=-2)
            combined = combined.squeeze()
        else:
            indices = torch.triu_indices(100,100, offset=1)
            ind_x = indices[0]
            ind_y = indices[1]
            combined = combined[...,ind_x,ind_y]

        self.dropout(combined)
        similarity = self.similarity_layer(combined)

        return similarity