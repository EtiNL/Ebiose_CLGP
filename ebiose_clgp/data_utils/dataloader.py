from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
from torch_geometric.data import Batch, Data

def get_dataloader(config, dataset, is_train=True):
    sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
    batch_size = config.train_batch_size * max(1, config.n_gpu) if is_train else config.per_gpu_eval_batch_size
    
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=config.num_workers, collate_fn=collate_training_data, pin_memory=True, prefetch_factor=2)



def collate_training_data(batch):
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