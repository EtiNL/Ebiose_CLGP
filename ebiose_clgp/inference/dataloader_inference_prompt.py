from torch.utils.data import DataLoader, SequentialSampler

def prompt_dataloader(config, dataset):
    sampler = SequentialSampler(dataset)
    batch_size = config.inference_batch_size * max(1, config.n_gpu)
    
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=config.num_workers, pin_memory=True, prefetch_factor=2)
