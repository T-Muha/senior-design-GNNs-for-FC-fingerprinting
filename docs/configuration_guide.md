# Configuration Guide


## Pipeline Configuration
* 'notes': allows you to record any notes about this execution of the pipeline since configuration is saved
* 'num_runs': number of runs to execute or, if cross-validation is used, the number of folds
* 'cross_validation': True if cross-validation is to be used instead of multiple independent runs


## Data

* 'dataset_type':
  * 'FC': Use FC data for dataset
  * 'all_random': Use a randomly-generated FC for each FC in each pair
  * 'scan_random': Generate random FCs for each subject, then construct pairs
* 'num_nodes': number of nodes to use for either randomly generated FCs or in the case of multiple FC datasets with different numbers of nodes
* 'full_dataset_size': dataset size before splitting
* 'valid_ratio': validation set ratio. Proportion of full dataset if not using cross-validation. Proportion of full dataset minus test size if using cross validation
* 'test_ratio': ratio of total data to use for testing, if not using cross-validation

## FC Preprocessing
* 'negative_edges': False if negative FC edges should be discarded
* 'greatest_k_edges': greatest k edges to keep for every FC
* 'min_edge_weight': the minimum edge magnitude to keep


## Training

### Batching
* 'batch_size': the batch size for training. Size of batch to put on device if effective batch size is used
* 'effective_batch_size': if greater than 'batch_size', use this effective batch size by accumulating the gradient

### Gradient Clipping
* 'clip_by': no clipping if 'none', clip at value if 'value', clip at norm if 'norm'
* 'clip_at': gradient value or norm to clip at


### Optimizer
* 'optimizer': either 'Adam' or 'AdamW'
* 'beta_1': first beta of the Adam optimizer. Default is 0.9
* 'beta_2': second beta of the Adam optimizer. Default is 0.999
* 'learning_rate': initial learning rate to use for training
* 'decrease_lr': True if learning rate should be decreased once a target loss is achieved
* 'decrease_lr_at': loss value at which to decrease the learning rate
* 'secondary_lr': the decreased learning rate if applicable



## Evaluation
* 'batch_size': batch size to use for evaluation in case not all examples fit in memory
* 'epochs': number of training epochs. Same across all runs
* 'margin': margin size for the hinge loss functions
* 'save_params': True if model parameters should be saved at end of every epoch. Can use a lot of memory for large models


## Model

* 'type': the model to use, see [the list of models](/docs/pipeline_details.md)
    * 'gmn'
    * 's_gcn'
    * 'hs_gcn'
    * 'mlp_embedding'
    * 'mlp_similarity'
    * 'mlp_diff_similarity'

### GMN
* 'num_prop_layers': number of propagation layers
* 'use_encoder': whether to encode the node and edge features
* 'node_feat_dim': the original dimensionality of node features
* 'edge_feat_dim': the original dimensionality of edge features
* 'node_hidden_dim': size of hidden layer in node encoder
* 'edge_hidden_dim': size of hidden layer in edge encoder
* 'node_state_dim': dimensionality of node state and size of output layer in the node encoder
* 'edge_state_dim': dimensionality of encoded edge and size of output layer in the edge encoder
* 'message_hidden_dim': size of the hidden layer in the message MLP
* 'message_dim': dimensionality of the messages and size of the output layer in the message MLP
* 'update_hidden_dim': size of the hidden later in the node update MLP
* 'graph_hidden_dim': size of the hidden layer in the graph embedding MLP
* 'graph_state_dim': dimensionality of the graph embedding and size of the output layer of the graph embedding MLP

### S_GCN
* 'knn_edges': number of greatest edges to keep in the knn graph
* 'layer_sizes': array of layer sizes for the spectal filtering layers in the GCNs

### HS_GCN
* 'num_walks': number of random walks at each node when constructing the co-occurance frequency matrix
* 'walk_len': length of the random walks
* 'window_size': size of the window to use when recording co-occurance frequency
