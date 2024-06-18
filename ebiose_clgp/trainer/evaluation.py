import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import hashlib

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

            graph_features, text_features = model(graphs, texts)

            # Unbatch graphs and texts
            graph_data_list = unbatch_graphs(graphs)
            texts_list = torch.split(texts, 1, dim=0)

            for graph_data, graph_feature, text, text_feature in zip(graph_data_list, graph_features, texts_list, text_features):
                graph_tensor = torch.cat([graph_data.x, graph_data.edge_index], dim=0).cpu()
                graph_hash = hash_tensor(graph_tensor)
                text_hash = hash_tensor(text.squeeze(0).cpu())

                graph_embeddings_map[graph_hash] = graph_feature.cpu().numpy()
                text_embeddings_map[text_hash] = text_feature.cpu().numpy()
    
    return graph_embeddings_map, text_embeddings_map

def calculate_similarity(train_embeddings, test_embeddings):
    similarities = cosine_similarity(test_embeddings, train_embeddings)
    return similarities

def evaluate_similarity(train_dataloader, test_dataloader, model, device='cuda'):
    model = model.to(device)
    
    train_graph_embeddings_map, train_text_embeddings_map = get_embeddings(model, train_dataloader, device)
    
    # Get graph and prompt embeddings from test dataset
    test_graph_embeddings_map, test_text_embeddings_map = get_embeddings(model, test_dataloader, device)
    
    # Calculate similarity between test prompt embeddings and train graph embeddings
    similarity_scores = calculate_similarity(train_graph_embeddings, test_text_embeddings)
    
    return similarity_scores

def model_eval(model, train_dataset, eval_dataset, test_dataset):
    
    # Evaluate similarity
    similarity_scores = evaluate_similarity(train_dataset, test_dataset, model)
    
    print(similarity_scores)

if __name__ == "__main__":
    main()
