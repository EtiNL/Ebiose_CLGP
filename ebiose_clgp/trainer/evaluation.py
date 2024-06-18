import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import hashlib
import wandb

def hash_tensor(tensor):
    """Generate a hash for a given tensor."""
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def unbatch_graphs(batch):
    data_list = batch.to_data_list()
    return data_list

def get_embeddings(model, dataloader, device):
    model.eval()
    graph_embeddings_map = {}
    text_embeddings_map = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            graphs, texts = batch
            graphs = graphs.to(device)
            texts = texts.to(device)

            graph_embeddings, text_embeddings = model(graphs, texts)

            # Unbatch graphs and texts
            graph_data_list = unbatch_graphs(graphs)
            texts_list = torch.split(texts, 1, dim=0)

            for graph_data, graph_embedding, text, text_embedding in zip(graph_data_list, graph_embeddings, texts_list, text_embeddings):
                graph_tensor = graph_data.x.cpu()
                graph_hash = hash_tensor(graph_tensor)
                text_hash = hash_tensor(text.cpu())

                graph_embeddings_map[graph_hash] = graph_embedding.cpu().numpy()
                text_embeddings_map[text_hash] = text_embedding.cpu().numpy()
    
    return graph_embeddings_map, text_embeddings_map

def evaluate_similarity(train_dataloader, test_dataloader, model, index_map, evaluation_map, hist_bins, device='cuda'):
    model = model.to(device)
    
    train_graph_embeddings_map, train_text_embeddings_map = get_embeddings(model, train_dataloader, device)
    test_graph_embeddings_map, test_text_embeddings_map = get_embeddings(model, test_dataloader, device)
    
    train_graph_hashes = set(train_graph_embeddings_map.keys())
    
    histogram_1 = []
    histogram_2 = []
    histogram_3 = []
    histogram_4 = []
    
    for idx, (graph_hash, prompt_hash) in index_map.items():
        eval_score = evaluation_map[(graph_hash, prompt_hash)]
        if graph_hash in test_graph_embeddings_map and prompt_hash in test_text_embeddings_map:
            test_graph_embedding = test_graph_embeddings_map[graph_hash]
            test_prompt_embedding = test_text_embeddings_map[prompt_hash]

            if graph_hash in train_graph_hashes:
                if eval_score:
                    print(1)
                    histogram_1.append(cosine_similarity([test_graph_embedding], [test_prompt_embedding])[0][0])
                else:
                    print(2)
                    histogram_2.append(cosine_similarity([test_graph_embedding], [test_prompt_embedding])[0][0])
            else:
                if eval_score:
                    print(3)
                    histogram_3.append(cosine_similarity([test_graph_embedding], [test_prompt_embedding])[0][0])
                else:
                    print(4)
                    histogram_4.append(cosine_similarity([test_graph_embedding], [test_prompt_embedding])[0][0])
    
    # Log histograms to wandb
    wandb.log({"known graph association test, if eval = true": wandb.Histogram(np_histogram=np.histogram(histogram_1, bins= hist_bins))})
    wandb.log({"known graph association test, if eval = false": wandb.Histogram(np_histogram=np.histogram(histogram_2, bins= hist_bins))})
    wandb.log({"generation metric test, if eval = true": wandb.Histogram(np_histogram=np.histogram(histogram_3, bins= hist_bins))})
    wandb.log({"generation metric test, if eval = false": wandb.Histogram(np_histogram=np.histogram(histogram_4, bins= hist_bins))})
    
    return histogram_1, histogram_2, histogram_3, histogram_4

