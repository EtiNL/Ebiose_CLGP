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
    
    # Combine node features and edge indices into a single batch
    graph_list = []
    for graph in graphs:
        node_features, edge_index = graph
        graph_list.append(Data(x=node_features, edge_index=edge_index))
    
    print('node_features.shape: ', node_features.shape)
    
    combined_graph = Batch.from_data_list(graph_list)
    
    print('collate_graph func x shape: ', (combined_graph.x).shape)
    
    # Stack text inputs
    texts = torch.stack(texts)
    
    return combined_graph, texts