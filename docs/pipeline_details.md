# Pipeline Details
A variety of models, data, and other features are available in the pipeline.

## Models

### Graph Matching Networks
An implemenation of the GMN described in "Graph Matching Networks for Learning the Similarity
of Graph Structured Objects." Embeds each graph in the given pair. Consists of two graph embedding networks with
a cross-graph attention mechanism in the propagation layers.

### Siamese Graph Convolutional Networks
An implementation of the S-GCN described in "Deep Graph Similarity Learning for Brain Data
Analysis." Returns a single similarity value for the given pair of graphs. Each graph is processed by one of the Siamese
graph convolutional networks, the outputs of which are combined with a dot product layer and mapped to a similarity value in the
similarity layer.

### Higher-Order Siamese Graph Convolutional Networks
An implementation of the HS-GCN described in "Deep Graph Similarity Learning for Brain Data
Analysis." Same as the S-GCN except a co-occurance frequency matrix is used as the knn-graph for the spectral filter layers in the
graph convolutional networks.

### Embedding Multilayer Perceptron
A MLP that takes a pair of graphs as adjacency matrices and independently embeds each

### Embedding "Difference" Multilayer Perceptron
Same as the Embedding MLP, except the difference between the two graph's adjacency matrix is used

### Similarity Multilayer Perceptron
A MLP that takes a pair of graphs as adjacency matrices and maps both to a similarity value


## Datasets

### FC Data
FCs are constructed using 1000 subjects from the HCP 1200 PTN dataset. The main dataset is built using the 6000 possible same-subject pairs and 6000 different-subject pairs. Each subject is uniformly sampled for the different-subject dataset.

### All random
FCs consist of randomly generated graphs. A new pair of random graphs is generated for every same-subject and every dif"ferent-subject pair.

### Scan random
FCs consist of randomly generated graphs. 4000 random graphs are generated, and the sampling used with the FC data is utilized to simulate sampling from 1000 subjects with 4 FCs each to construct the same-subject and different-subject pairs.


## Run Modes

### Multiple Runs
One or more runs are executed using the given configuration, with a new dataset constructed at the beginning of each run.

### Cross-Validation
K-Fold Cross-Validation is used for evaluation. Only one run of cross-validation can be executed. Number of folds is specified through the "num_runs" config value.

