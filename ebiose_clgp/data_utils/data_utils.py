import torch
from torch_geometric.data import Batch, Data

def collate_graph(batch):
    graphs, texts, labels = zip(*batch)
    
    # Combine node features and edge indices into a single batch
    graph_list = []
    for graph in graphs:
        node_features, edge_index = graph
        graph_list.append(Data(x=node_features, edge_index=edge_index))
    
    # print('node_features.shape: ', node_features.shape)
    
    combined_graph = Batch.from_data_list(graph_list)
    
    # print('collate_graph func x shape: ', (combined_graph.x).shape)
    
    # Stack text inputs
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    
    return combined_graph, texts, labels