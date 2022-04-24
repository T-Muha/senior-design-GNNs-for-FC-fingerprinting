import numpy as np
import torch
import os
import sys
from run_pipeline import run_pipeline

config = {}

# Determine available resources
try:
    os.environ['COLAB_TPU_ADDR']
    print("Using TPU")
    config['TPU'] = True
except:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    torch.autograd.set_detect_anomaly(True)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print("%i Device(s) Available." % torch.cuda.device_count(), end=' ')
    print("Using Device {}".format(device))
    config['TPU'] = False

# For more readable numpy arrays
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(edgeitems=30, linewidth=100000,
    formatter=dict(float=lambda x: "%.2g" % x))

# Set random seeds
# seed = 10
# random.seed(seed)
# np.random.seed(seed + 1)
# torch.manual_seed(seed + 2)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

config['notes'] = ""

config['num_runs'] = 5

config['train'] = {}
config['train']['batch_size'] = 200
config['train']['effective_batch_size'] = 400
config['train']['epochs'] = 200
config['train']['margin'] = 1.0

config['train']['save_params'] = False

config['evaluation'] = {}
config['evaluation']['batch_size'] = 200

config['model'] = {}

config['model']['type'] = 's_gcn'

config['model']['layer_sizes'] = [16, 16]

# config['model']['use_encoder'] = False
# config['model']['node_feat_dim'] = 8
# config['model']['edge_feat_dim'] = 1

# config['model']['node_hidden_dim'] = None
# config['model']['node_state_dim'] = None
# config['model']['edge_hidden_dim'] = None
# config['model']['edge_encoded_dim'] = None

# config['model']['message_hidden_dim'] = 32
# config['model']['message_dim'] = 4

# config['model']['update_hidden_dim'] = 32

# config['model']['graph_hidden_dim'] = 16
# config['model']['graph_state_dim'] = 16
# config['model']['num_prop_layers'] = 3

config['model']['knn_edges'] = 10

config['model']['random_walk'] = {}
config['model']['random_walk']['num_walks'] = 10
config['model']['random_walk']['walk_len'] = 50
config['model']['random_walk']['window_size'] = 5

config['dataset_type'] = 'FC'

config['data'] = {}
config['data']['num_subjects'] = 1000
config['data']['num_days'] = 4
config['data']['num_nodes'] = 100
config['data']['full_size'] = 12000
config['data']['valid_ratio'] = 0.10
config['data']['test_ratio'] = 0.10

config['data']['negative_edges'] = False
config['data']['greatest_k_edges'] = 200
config['data']['min_edge_weight'] = 0.2

config['train']['grad_clip'] = {}
config['train']['grad_clip']['clip_by'] = 'none'
config['train']['grad_clip']['clip_at'] = 50

config['train']['optimizer'] = 'AdamW'
config['train']['weight_decay'] = 1e-5
config['train']['beta_1'] = 0.9
config['train']['beta_2'] = 0.999

config['train']['learning_rate'] = 1e-4
config['train']['decrease_lr'] = True
config['train']['decrease_lr_at'] = 0.70
config['train']['secondary_lr'] = 1e-4

config['cross_validation'] = True

run_pipeline(config, device)