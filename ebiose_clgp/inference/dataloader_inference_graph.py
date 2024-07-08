from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Batch, Data

def prompt_dataloader(config, dataset):
    sampler = SequentialSampler(dataset)
    batch_size = config.inference_batch_size * max(1, config.n_gpu)
    
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=config.num_workers, collate_fn=collate_graphs, pin_memory=True, prefetch_factor=2)

def collate_graphs(batch):
# Combine node features and edge indices into a single batch
    graph_list = []
    for graph in batch:
        node_features, edge_index = graph
        graph_list.append(Data(x=node_features, edge_index=edge_index))
    
    # print('node_features.shape: ', node_features.shape)
    
    combined_graph = Batch.from_data_list(graph_list)
    
    return combined_graph