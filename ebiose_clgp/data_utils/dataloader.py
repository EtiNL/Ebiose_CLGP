from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from ebiose_clgp.data_utils.data_utils import collate_graph

def get_dataloader(config, dataset, is_train=True):
    sampler = RandomSampler(dataset) if is_train else SequentialSampler(dataset)
    batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)
    
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=config.num_workers, collate_fn=collate_graph, pin_memory=True, prefetch_factor=2)
