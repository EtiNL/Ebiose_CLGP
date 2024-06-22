import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

# Node Feature Perturbation
def perturb_node_features(data, noise_factor=0.01):
    noise = torch.randn_like(data.x) * noise_factor
    data.x += noise
    return data

# Edge Perturbation
def perturb_edges(data, add_edges=5, remove_edges=5):
    num_nodes = data.num_nodes

    # Remove random edges
    for _ in range(remove_edges):
        if data.edge_index.size(1) > 0:
            edge_index = data.edge_index[:, torch.randperm(data.edge_index.size(1))]
            data.edge_index = edge_index[:, :-1]

    # Add random edges
    for _ in range(add_edges):
        new_edge = torch.randint(0, num_nodes, (2,))
        data.edge_index = torch.cat([data.edge_index, new_edge.unsqueeze(1)], dim=1)

    return data

# Subgraph Extraction
def extract_subgraph(data, subgraph_size=10):
    num_nodes = data.num_nodes
    subgraph_nodes = torch.randperm(num_nodes)[:subgraph_size]
    subgraph_edge_index, _ = subgraph(data.edge_index, subset=subgraph_nodes, relabel_nodes=True)
    subgraph_data = Data(x=data.x[subgraph_nodes], edge_index=subgraph_edge_index)
    return subgraph_data

# Graph Augmentation with Noise
def add_noise_to_graph(data, noise_factor=0.01):
    data = perturb_node_features(data, noise_factor)
    data = perturb_edges(data)
    return data

