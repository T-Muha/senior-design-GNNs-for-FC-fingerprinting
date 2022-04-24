import torch

def euclidean_distance(g_1, g_2):
    """Compute Euclidean Distance for batch of graph pairs."""

    distance = torch.sum((g_1 - g_2) ** 2, axis=-1)
    return distance


def similarity_loss(similarities, labels, margin):
    """Returns loss for a batch of similarity scores and labels."""

    return torch.sum(torch.relu(margin - labels * similarities.squeeze())) / similarities.shape[0]


def embedding_loss(g_1, g_2, labels, margin):
    """Returns loss for batches of graph embeddings using given labels."""
    
    distance = euclidean_distance(g_1, g_2)
    loss = torch.sum(torch.relu(margin - labels * (1 - distance)))
    return loss / g_1.shape[0]