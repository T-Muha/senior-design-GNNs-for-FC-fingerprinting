import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np

from utils.graph_utils import get_features
from loss import similarity_loss, embedding_loss

def train(device, model, optimizer, train_pairs_in, train_labels_in, config):
    """Trains model over one pass of the dataset using the given optimizer."""

    model.train()

    indices = np.random.permutation(len(train_pairs_in))
    train_pairs = train_pairs_in[indices]
    train_labels = train_labels_in[indices]

    accum_grad_norm = []

    batch_size = config['train']['batch_size']
    effective_batch_size = config['train']['effective_batch_size']
    model_type = config['model']['type']
    batch_start = 0
    batch_end = batch_size

    if effective_batch_size > batch_size:
        gradient_accumulation = True
        accumulated_batches = 0
    else:
        gradient_accumulation = False

    # Do not use partial batches, they will have a disproportional affect on training
    while batch_end <= len(train_pairs):

        batch_x = torch.from_numpy(train_pairs[batch_start:batch_end]).float()
        batch_labels = torch.from_numpy(train_labels[batch_start:batch_end])

        batch_start += batch_size
        batch_end += batch_size

        classes = torch.unique(batch_labels)
        if not 1 in classes:
            print("No positive examples in batch!")
        if not -1 in classes:
            print("No negative examples in batch")

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
        
        # Embedding models
        else:
            if model_type == 'mlp_embedding':
                graph_vectors = model(batch_x.to(device))
                graph_vect_1, graph_vect_2 = torch.unbind(graph_vectors, dim=1)
            elif model_type == 'gmn':
                graph_vectors = model(device, node_features.to(device), edge_features.to(device), edge_vertices.to(device))
                graph_vect_1, graph_vect_2 = torch.unbind(graph_vectors, dim=1)
            batch_loss = embedding_loss(graph_vect_1, graph_vect_2, batch_labels.to(device), config['train']['margin'])

        if gradient_accumulation:
            accumulated_batches += batch_size
            if accumulated_batches == batch_size:
                # First batch of gradient accumulation
                optimizer.zero_grad()
                batch_loss.backward()
            elif accumulated_batches < effective_batch_size:
                batch_loss.backward()
            else:
                # Last batch of gradient accumulation
                batch_loss.backward()
                if config['train']['grad_clip']['clip_by'] == 'value':
                    clip_grad_value_(model.parameters(), config['train']['grad_clip']['clip_at'])
                elif config['train']['grad_clip']['clip_by'] == 'norm':
                    clip_grad_norm_(model.parameters(), config['train']['grad_clip']['clip_at'])
                optimizer.step()
                accumulated_batches = 0

                grad_norm = 0
                params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                for p in params:
                    param_norm = p.grad.detach().data.norm(2)
                    grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** 0.5

                accum_grad_norm.append(grad_norm)

        else:
            optimizer.zero_grad()
            batch_loss.backward()
            if config['train']['grad_clip']['clip_by'] == 'value':
                clip_grad_value_(model.parameters(), config['train']['grad_clip']['clip_at'])
            elif config['train']['grad_clip']['clip_by'] == 'norm':
                clip_grad_norm_(model.parameters(), config['train']['grad_clip']['clip_at'])
            optimizer.step()

            grad_norm = 0
            params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in params:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5

            accum_grad_norm.append(grad_norm)

    gradient_norm = np.mean(accum_grad_norm)

    return gradient_norm