import torch
from torch_geometric.data import Batch, Data

def pad_graphs(graphs):
    max_nodes = max([graph[0].size(0) for graph in graphs])
    padded_graphs = []

    for node_features, edge_index in graphs:
        pad_size = max_nodes - node_features.size(0)
        if pad_size > 0:
            pad_tensor = torch.zeros((pad_size, node_features.size(1)), dtype=node_features.dtype)
            node_features = torch.cat([node_features, pad_tensor], dim=0)
        
        padded_graphs.append((node_features, edge_index))

    return padded_graphs

def collate_graph(batch):
    graphs, texts = zip(*batch)
    graphs = pad_graphs(graphs)
    node_features_list, edge_index_list = zip(*graphs)

    # Use torch_geometric's Batch to combine node features and edge indices into a batch
    combined_graph = Batch.from_data_list([Data(x=node_features, edge_index=edge_index) for node_features, edge_index in graphs])

    # Stack text inputs
    texts = torch.stack(texts)

    return combined_graph, texts