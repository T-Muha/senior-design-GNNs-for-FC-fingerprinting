# Pipeline Structure
This document outlines the file structure of the pipeline.

**main.py**  
Contains pipeline configuration and runs the pipeline. Execute this file to run the pipeline with the given configuration.

**run_pipeline.py**  
Executes the pipeline using the configuration passed to it by main.py.

**FC_graphs.py**  
Functions for creating FCs and building the same-subject/different-subject pairs.

**random_FC.py**  
Contains functions for generating random FCs and random-FC datasets.

**split_dataset**  
Contains functions for splitting the dataset into train/validation/test sets for both normal and
cross-validation contexts.

**create_model.py**  
Creates network and optimizer from the given configuration.

**train.py**  
Contains the function that trains the given model using the given training set.

**evaluate.py**  
Contains the function that evaluates the given model using the given dataset.

**loss.py**  
Contains loss functions for both similarity values and pairs of embeddings.

**metrics.py**  
Contains functions for calculating the accuracy and AUC metrics for both similarity values and pairs of embeddings.

### /models

- **MLP_similarity.py**  
MLP that predicts similarity of a pair of graphs by mapping them to a similarity value.

- **MLP_diff_similarity.py**  
MLP that predicts the similarity of a pair of graphs by mapping their difference to a similarity value.

- **MLP_embedding.py**  
MLP that independently embeds a pair of graphs.

- **GMN.py**  
Implements the Graph Matching Network described in "Graph Matching Networks for Learning the Similarity of Graph Structured Objects."

- **S_GCN.py**  
Implements the Siamese Graph Convolutional Network described in Deep Graph Similarity Learning for Brain Data Analysis." The Higher-Order Siamese Graph Convolutional Network is instead used if specified in the pipeline configuration.

### /utils
- **graph_utils.py**  
Contains graph-related utilities: knn-graph creation, random walks, co-occurence frequency generation,
and graph node/edge feature extraction.

- **plot.py**  
Contains functions for saving data plots.

- **results.py**  
Contains functions related to recording pipeline results, including recording model parameter values.