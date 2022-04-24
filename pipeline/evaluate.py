import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from metrics import similarity_accuracy, similarity_auc, embedding_accuracy, embedding_auc
from loss import similarity_loss, embedding_loss
from utils.graph_utils import get_features

def evaluate(device, model, pairs, labels, config, shuffle=False):
    """Evaluate on validation or test data."""

    if shuffle:
        indices = np.random.permutation(len(pairs))
        pairs = pairs[indices]
        labels = labels[indices]

    model_type = config['model']['type']
    batch_size = config['evaluation']['batch_size']

    accum_acc = []
    accum_loss = []
    accum_auc = []
    accum_in_margin = []
    accum_distances = np.array([])
    correct = np.array([])

    model.eval()

    batch_start = 0
    batch_end = batch_size

    with torch.no_grad():
        while batch_start < len(pairs):

            if batch_end > len(pairs):
                batch_x = torch.from_numpy(pairs[batch_start:]).float()
                batch_labels = torch.from_numpy(labels[batch_start:])
            else:
                batch_x = torch.from_numpy(pairs[batch_start:batch_end]).float()
                batch_labels = torch.from_numpy(labels[batch_start:batch_end])

            batch_start += batch_size
            batch_end += batch_size

            # Get node and edge features
            if model_type == 'gmn':
                node_features, edge_features, edge_vertices = get_features(batch_x, config['model']['node_feat_dim'], config['model']['edge_feat_dim'])

            # Similarity models
            if model_type in 's_gcn | hs_gcn | mlp_similarity | mlp_diff_similarity':

                if model_type in 's_gcn | hs_gcn':
                    similarities = model(device, batch_x.to(device))
                elif model_type == 'mlp_similarity':
                    similarities = model(batch_x.to(device))
                elif model_type == 'mlp_diff_similarity':
                    batch_x_diff = batch_x[:,0,:,:] - batch_x[:,1,:,:]
                    similarities = model(batch_x_diff.to(device))
            
                batch_loss = similarity_loss(similarities, batch_labels.to(device), config['train']['margin'])    
                batch_acc, batch_correct, batch_in_margin = similarity_accuracy(similarities, batch_labels.to(device), config['train']['margin'])
                batch_auc = similarity_auc(similarities, batch_labels.to(device))
                distances = similarities

            # Embedding models
            else:
                if config['model']['type'] == 'mlp_embedding':
                    graph_vectors = model(batch_x.to(device))
                    graph_vect_1, graph_vect_2 = torch.unbind(graph_vectors, dim=1)
                elif config['model']['type'] == 'gmn':
                    graph_vectors = model(device, node_features.to(device), edge_features.to(device), edge_vertices.to(device))
                    graph_vect_1, graph_vect_2 = torch.unbind(graph_vectors, dim=1)

                batch_loss = embedding_loss(graph_vect_1, graph_vect_2, batch_labels.to(device), None, config['train']['margin'])
                batch_acc, distances, batch_correct, batch_in_margin = embedding_accuracy(graph_vect_1, graph_vect_2, batch_labels.to(device), config['train']['margin'])
                batch_auc = embedding_auc(graph_vect_1, graph_vect_2, batch_labels.to(device))

            accum_loss.append(batch_loss.cpu())
            accum_acc.append(batch_acc.cpu())
            accum_auc.append(batch_auc)
            accum_distances = np.append(accum_distances, distances.cpu())
            accum_in_margin.append(batch_in_margin.cpu())
            correct = np.append(correct, batch_correct.cpu().numpy(), axis=0)
        
    avg_loss = np.mean(accum_loss)
    avg_acc = np.mean(accum_acc)
    avg_auc = np.mean(accum_auc)
    avg_in_margin = np.mean(accum_in_margin)

    accum_distances = np.array(accum_distances)
    accum_distances = np.expand_dims(accum_distances, axis = 1)
    correct = np.expand_dims(correct, axis=1)

    if model_type in 's_gcn | hs_gcn | mlp_similarity':
        conf_matrix = confusion_matrix(labels, np.sign(accum_distances))
    else:
        conf_matrix = confusion_matrix(labels, np.sign(1 - accum_distances))

    return avg_loss, avg_acc, avg_auc, accum_distances, correct, avg_in_margin, conf_matrix