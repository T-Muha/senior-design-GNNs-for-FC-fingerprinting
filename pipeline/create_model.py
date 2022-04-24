from models.MLP_embedding import MLP_Embedding
from models.MLP_similarity import MLP_Similarity
from models.MLP_diff_similarity import MLP_Diff_Similarity
from models.GMN import GMN
from models.S_GCN import S_GCN
from torch.optim import Adam, AdamW


def create_model(config, knn_G):
    """
    Builds and returns the model and optimizer.
    knn_G: an average knn graph necessary for gcn-based models
    """

    params = config['model']
    modelname = config['model']['type']

    if modelname in 's_gcn | hs_gcn':
        model = S_GCN(config['model']['layer_sizes'], knn_G)
    elif modelname == 'gmn':
        model = GMN(config['data']['num_nodes'], params['node_feat_dim'], params['node_hidden_dim'], params['node_state_dim'],
                    params['edge_feat_dim'], params['edge_hidden_dim'], params['edge_encoded_dim'],
                    params['message_hidden_dim'], params['message_dim'], params['update_hidden_dim'],
                    params['graph_hidden_dim'], params['graph_state_dim'],
                    params['num_prop_layers'], params['use_encoder'])
    elif modelname == 'mlp_embedding':
        model = MLP_Embedding()
    elif modelname == 'mlp_similarity':
        model = MLP_Similarity()
    elif modelname == 'mlp_diff_similarity':
        model = MLP_Diff_Similarity()

    # Build optimizer
    betas = (config['train']['beta_1'], config['train']['beta_2'])
    if config['train']['optimizer'] == 'Adam':
        optimizer = Adam((model.parameters()), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'], betas=betas)
    elif config['train']['optimizer'] == 'AdamW':
        optimizer = AdamW((model.parameters()), lr=config['train']['learning_rate'], weight_decay=config['train']['weight_decay'], betas=betas)
    
    return model, optimizer