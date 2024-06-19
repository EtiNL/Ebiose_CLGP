import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import hashlib
import wandb
import os
import pickle
import zipfile

def hash_tensor(tensor):
    """Generate a hash for a given tensor."""
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

def unbatch_graphs(batch):
    data_list = batch.to_data_list()
    return data_list

def save_embeddings_map(path, embedding_maps):
    with open(path, 'wb') as f:
        pickle.dump(embedding_maps, f)

def load_embeddings_map(path):
    if path.endswith('.zip'):
        print('unzipping dataset...')
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(path))
        path = path.rstrip('.zip')+'.pkl' # Update the path to the unzipped file
        print("done")
    with open(path, 'rb') as f:
        return pickle.load(f)

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

def evaluate_similarity(train_dataloader, test_dataloader, model, index_map, evaluation_map, hist_bins, device='cuda', saving_path=None):
    model = model.to(device)
    
    if saving_path and os.path.exists(saving_path):
        train_graph_embeddings_map, train_text_embeddings_map, test_graph_embeddings_map, test_text_embeddings_map = load_embeddings_map(saving_path)
    else:
        train_graph_embeddings_map, train_text_embeddings_map = get_embeddings(model, train_dataloader, device)
        test_graph_embeddings_map, test_text_embeddings_map = get_embeddings(model, test_dataloader, device)
        
        if saving_path:
            save_embeddings_map(saving_path, (train_graph_embeddings_map, train_text_embeddings_map, test_graph_embeddings_map, test_text_embeddings_map))
    
    train_graph_hashes = set(train_graph_embeddings_map.keys())
    test_graph_hashes = set(train_graph_embeddings_map.keys())
    test_prompt_hashes = set(test_text_embeddings_map.keys())

    histogram_1 = []
    histogram_2 = []
    histogram_3 = []
    histogram_4 = []
    
    print(len(train_graph_hashes), len(test_graph_hashes))

    for (graph_hash, prompt_hash) in index_map.values():
        # print(graph_hash in test_graph_hashes or graph_hash in train_graph_hashes or prompt_hash in test_prompt_hashes)
        eval_score = evaluation_map[(graph_hash, prompt_hash)]
        if graph_hash in test_graph_hashes and prompt_hash in test_prompt_hashes:
            print(True)
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

    breakpoint()
    # Log histograms to wandb
    wandb.log({"known graph association test, if eval = true": wandb.Histogram(np_histogram=np.histogram(histogram_1, bins=hist_bins))})
    wandb.log({"known graph association test, if eval = false": wandb.Histogram(np_histogram=np.histogram(histogram_2, bins=hist_bins))})
    wandb.log({"generation metric test, if eval = true": wandb.Histogram(np_histogram=np.histogram(histogram_3, bins=hist_bins))})
    wandb.log({"generation metric test, if eval = false": wandb.Histogram(np_histogram=np.histogram(histogram_4, bins=hist_bins))})

    return histogram_1, histogram_2, histogram_3, histogram_4

if __name__ == "__main__":
    from ebiose_clgp.trainer.utils import mkdir, load_config_file
    from ebiose_clgp.data_utils.tokenizer import get_max_position_embedding
    from omegaconf import OmegaConf
    from ebiose_clgp.data_utils.dataloader import get_dataloader
    from ebiose_clgp.data_utils.dataset import CLGP_Ebiose_dataset
    from ebiose_clgp.model.CLGP import CLGP
    from ebiose_clgp.trainer.train_utils import set_seed
    from ebiose_clgp.trainer.evaluation import evaluate_similarity
    from ebiose_clgp.model.text_encoders.bert import get_Bert
    
    DATA_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/data_utils/data_config.yaml'
    TRAINER_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/trainer/bert_train_config.yaml'
    MODEL_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/model/bert_model_config.yaml'
    # Load your config, model, and dataloaders
    data_config = load_config_file(DATA_CONFIG_PATH)
    train_config = load_config_file(TRAINER_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)

    config = OmegaConf.merge(train_config, data_config, model_config)

    wandb.init(project='Ebiose_CLGP')

    mkdir(path=config.saved_checkpoints)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count()
    
    if config.text_encoder.name == 'Transformer':
        config.graph_node_tokenizer_max_pos = get_max_position_embedding(config.graph_feature_tokenizer)
        config.prompt_tokenizer_max_pos = get_max_position_embedding(config.prompt_tokenizer)
        tokenizer = None
        model = None
        
    elif config.text_encoder.name == 'Bert':
        tokenizer, model = get_Bert()
        config.embed_dim = 768 #Bert embedding dimension
    
    config.model_save_name = wandb.run.name
    
    set_seed(seed=11, n_gpu=config.n_gpu)
    
    model = CLGP(config, model)

    # Load the model from the saved checkpoint
    checkpoint = torch.load(config.saved_model)
    if config.n_gpu > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    dataset = CLGP_Ebiose_dataset(config, tokenizer=tokenizer)
    train_dataset, val_dataset, test_dataset = dataset.train_validation_test_split()
    
    print("model evaluation...")
    # Evaluate similarity and log histograms
    train_dataloader = get_dataloader(config, train_dataset, is_train=False)
    test_dataloader = get_dataloader(config, test_dataset, is_train=False)
    evaluate_similarity(train_dataloader, test_dataloader, model, dataset.index_map, dataset.evaluation_map, config.eval_hist_bins, config.device, config.embbeddings_saving_path)
