import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import wandb
import os
import pickle
import zipfile
from ebiose_clgp.data_utils.dataloader import get_dataloader

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


def evaluate_similarity(test_dataset, model, hist_bins, title, device='cuda', saving_path=None):
    
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

            for graph_embedding, text_embedding, label in zip(graph_embeddings, text_embeddings, labels):
                if label == True:
                    hist_true.append(cosine_similarity([graph_embedding], [text_embedding])[0][0])
                elif label == False:
                    hist_false.append(cosine_similarity([graph_embedding], [text_embedding])[0][0])
                else:
                    raise Exception(f'type label: {type(label)}, instead of bool')
                    

    # Log histograms to wandb using wandb.Table and wandb.plot.histogram
    def log_histogram(data, title):
        table = wandb.Table(data=[[x] for x in data], columns=["value"])
        histogram = wandb.plot.histogram(table, value='cosine similarity', title=title)
        wandb.log({title: histogram})    

    log_histogram(hist_true, f"{title}, evaluation = 1")
    log_histogram(hist_false, f"{title}, evaluation = 0")

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
    test_dataloader = get_dataloader(config, test_dataset, is_train=False)
    evaluate_similarity(test_dataloader, model, config.eval_hist_bins, config.device, config.embbeddings_saving_path)
    print("done")
    wandb.finish()
