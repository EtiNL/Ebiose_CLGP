import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import wandb
import os
import pickle
import zipfile
from ebiose_clgp.data_utils.dataloader import get_dataloader

def save_embeddings(path, embedding_maps):
    with open(path, 'wb') as f:
        pickle.dump(embedding_maps, f)

def load_embeddings(path):
    if path.endswith('.zip'):
        print('unzipping dataset...')
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(path))
        path = path.rstrip('.zip')+'.pkl' # Update the path to the unzipped file
        print("done")
    with open(path, 'rb') as f:
        return pickle.load(f)

# Log histograms to wandb using wandb.Table and wandb.plot.histogram
def log_histogram(data, title):
    table = wandb.Table(data=[[x] for x in data], columns=["value"])
    histogram = wandb.plot.histogram(table, value='value', title=title)
    wandb.log({title: histogram})  


def evaluate_similarity(config, test_dataset, model, title):
    
    device = config.device
    saving_path = config.embbeddings_saving_path
    hist_bins = config.get('eval_hist_bins', 30)
    
    test_dataloader = get_dataloader(config, test_dataset, is_train=False)
    
    model = model.to(device)
    model.eval()
    
    hist_true = []
    hist_false = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Generating embeddings"):
            input_graphs, texts, labels = batch
            input_graphs = input_graphs.to(device)
            texts = texts.to(device)
            # # Unbatch graphs and texts
            # graph_data_list = unbatch_graphs(input_graphs)

            graph_embeddings, text_embeddings = model(input_graphs, texts)
            
            similarities = F.cosine_similarity(graph_embeddings, text_embeddings)

            for graph_embedding, text_embedding, label, similarity in zip(graph_embeddings, text_embeddings, labels, similarities):
                if label == 1:
                    hist_true.append(similarity)
                elif label == 0:
                    hist_false.append(similarity)
                else:
                    try:
                        print(f"label shape: {label.shape}")
                    except:
                        raise Exception(f'type label: {type(label)}, instead of int or array_like')  

    log_histogram(hist_true, f"{title}, evaluation = 1")
    log_histogram(hist_false, f"{title}, evaluation = 0")

if __name__ == "__main__":
    # needs to be updated
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
    test_dataloader = get_dataloader(config, test_dataset, is_train=False)
    evaluate_similarity(config, test_dataloader, model)
    print("done")
    wandb.finish()
