import numpy as np

def random_FC(num_nodes):
    """Generates an FC containing random data."""

    FC = np.random.normal(scale=0.1, size=(num_nodes, num_nodes)).clip(min=0, max=1)
    FC = np.triu(FC)
    FC = FC + FC.T
    np.fill_diagonal(FC, 0)
    return FC

def random_dataset(size, num_nodes):
    """
    Generates a dataset of same-subject and different-subject
    FC pairs using two new random FCs in each pair.
    """

    pairs = np.zeros((size, 2, num_nodes, num_nodes))
    for i in range(size):
        for j in range(2):
            pairs[i][j] = random_FC(num_nodes)
    labels = np.random.choice([1, -1], (size))
    return pairs, labels