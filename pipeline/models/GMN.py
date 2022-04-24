import torch
from torch import nn

from torch_scatter import scatter_add

def euclidean_distance_similarity(nodes_1, nodes_2):
    """
    Return pairwise euclidean distance similarity matrix between nodes_1 and nodes_2.
    nodes_1 = (N, D) tensor of nodes of first graph in pair.
    nodes_2 = (N, D) tensor of nodes of first graph in pair.
    ED(n1_i, n2_j) = -(n1_i - n2_j)^2 = 2*n1_i*n2_j - n1_i^2 - n2_j^2 = c - a - b
    """

    # Get squared nodes. Unsqueeze to get (1, N) shape
    a = torch.sum(nodes_1*nodes_1, dim=-1)
    b = torch.sum(nodes_2*nodes_2, dim=-1)
    a = torch.unsqueeze(a, 0)
    b = torch.unsqueeze(b, -1)
    c = 2 * torch.mm(nodes_1, torch.transpose(nodes_2, 0, 1))
    
    return c - a - b


def cross_attention_vectors(nodes_1, nodes_2):
    """
    Return two matrices containing cross-graph attention vectors for nodes_1 and nodes_2.
    nodes_1 = (N, D) tensor of nodes of first graph in pair.
    nodes_2 = (N, D) tensor of nodes of first graph in pair.
    """
    sim_matrix = euclidean_distance_similarity(nodes_1, nodes_2)
    a_weight_1 = torch.softmax(sim_matrix, dim=1)
    a_weight_2 = torch.transpose(torch.softmax(sim_matrix, dim=0), 0, 1)
    attention_vec_1 = nodes_1 - torch.mm(a_weight_1, nodes_2)
    attention_vec_2 = nodes_2 - torch.mm(a_weight_2, nodes_1)

    return attention_vec_1, attention_vec_2

class AttentionPropagationLayer(nn.Module):
    """A single graph propagation layer for the GMN. Also computes cross-graph attention."""

    def __init__(self, num_nodes, node_state_dim, edge_encoded_dim, message_hidden_dim, message_dim, update_hidden_dim):
        super(AttentionPropagationLayer, self).__init__()

        self.num_nodes = num_nodes
        self.node_state_dim = node_state_dim
        self.message_dim = message_dim

        self.message_encoder = nn.Sequential(
            nn.Linear(node_state_dim*2+edge_encoded_dim, message_hidden_dim),
            nn.ReLU(),
            nn.Linear(message_hidden_dim, message_hidden_dim),
            nn.ReLU(),
            nn.Linear(message_hidden_dim, message_dim)
        )

        self.node_update = nn.Sequential(
            nn.Linear(node_state_dim+message_dim+node_state_dim, update_hidden_dim),
            nn.ReLU(),
            nn.Linear(update_hidden_dim, update_hidden_dim),
            nn.ReLU(),
            nn.Linear(update_hidden_dim, node_state_dim)
        )

    def forward(self, device, node_states, edges, vertices):
        nodes_i = node_states[vertices[:,0]]
        nodes_j = node_states[vertices[:,1]]

        messages = self.message_encoder(torch.cat((nodes_i, nodes_j, edges), axis=1))

        # Augment messages with a zero-filled message for each node
        # Nodes without messages will have a zero for their sum
        messages_aug = torch.cat((messages.repeat(2,1), torch.zeros((len(node_states), self.message_dim), device=device)))

        # Sum every node's messages
        vertices_flat = torch.cat((vertices[:,0], vertices[:,1], torch.arange(0,len(node_states),device=device)))
        summed_messages = scatter_add(dim=0, index=vertices_flat, src=messages_aug)

        # Pair nodes by based on whether they are first or second graph in pair
        paired_nodes = node_states.view(-1, 2, self.num_nodes, self.node_state_dim)

        # Subtract corresponding nodes and return to original node_state shape
        attention_vectors = torch.cat((paired_nodes[:,0,:,:] - paired_nodes[:,1,:,:], paired_nodes[:,1,:,:] - paired_nodes[:,0,:,:]), axis=1).view(-1, self.node_state_dim)

        node_update_input = torch.cat((node_states, summed_messages, attention_vectors), axis=1)
        updated_node_states = self.node_update(node_update_input)

        return updated_node_states



class GMN(nn.Module):
    """
    GMN described in "Graph Matching Networks for Learning the Similarity
    of Graph Structured Objects." Embeds a pair of graphs to a pair of embedding vectors.
    In the propagation layers, messages are computed for each edge, and a node state is updated
    using its messages and a cross-graph attention vector. Node states are aggregated and mapped
    to an embedding.
    """

    def __init__(self, num_nodes, node_feat_dim, node_hidden_dim, node_state_dim,
                       edge_feat_dim, edge_hidden_dim, edge_encoded_dim,
                       message_hidden_dim, message_dim, update_hidden_dim,
                       graph_hidden_dim, graph_state_dim,
                       num_prop_layers, use_encoder):

        super(GMN, self).__init__()
        
        # Encoding not necessary in some cases
        self.use_encoder = use_encoder

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim

        if self.use_encoder:
            self.node_encoder = nn.Sequential(
                nn.Linear(self.node_feat_dim, node_hidden_dim),
                nn.ReLU(),
                nn.Linear(node_hidden_dim, self.node_state_dim)
            )
            self.edge_encoder = nn.Sequential(
                nn.Linear(self.edge_feat_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim, self.edge_encoded_dim)
            )
            self.node_state_dim = node_state_dim
            self.edge_encoded_dim = edge_encoded_dim
        else:
            # Encoded dimensions same as feature dimensions if no encoding performed
            self.node_state_dim =  node_feat_dim
            self.edge_encoded_dim = edge_feat_dim

        # Propagation layer parameters are shared
        self.num_prop_layers = num_prop_layers
        self.propagation_layer = AttentionPropagationLayer(num_nodes, self.node_state_dim, self.edge_encoded_dim, message_hidden_dim, message_dim, update_hidden_dim)

        self.node_gate_net = nn.Sequential(
            nn.Linear(self.node_state_dim, self.node_state_dim),
            nn.Sigmoid()
        )

        self.aggregator = nn.Sequential(
            nn.Linear(self.node_state_dim, graph_hidden_dim),
            nn.ReLU(),
            nn.Linear(graph_hidden_dim, graph_state_dim)
        )


    def forward(self, device, node_features, edge_features, edge_vertices):

        batch_size = node_features.shape[0]
        
        # Encode nodes, edges if desired
        node_states = self.node_encoder(node_features) if self.use_encoder else node_features
        edge_embedded = self.edge_encoder(edge_features) if self.use_encoder else edge_features

        node_states = torch.flatten(node_states, end_dim=-2)
        for _ in range(self.num_prop_layers):
            node_states = self.propagation_layer(device, node_states, edge_embedded, edge_vertices)
        
        # Aggregate nodes
        node_gates = self.node_gate_net(node_states)
        node_states = node_states * node_gates

        # Take batch/pair view of node states for per-graph summations
        batched_paired_node_states = node_states.view(batch_size,2,-1,self.node_state_dim)
        graph_states = torch.sum(batched_paired_node_states, dim=-2)
        graph_embedded = self.aggregator(graph_states)

        return graph_embedded