import torch
from omegaconf import OmegaConf
import wandb
from ebiose_clgp.trainer.utils import mkdir, load_config_file
from ebiose_clgp.data_utils.tokenizer import get_max_position_embedding
from ebiose_clgp.model.CLGP import CLGP
from ebiose_clgp.data_utils.dataset import CLGP_Ebiose_dataset
from ebiose_clgp.model.text_encoders.bert import get_Bert
from torch.utils.data import DataLoader

DATA_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/data_utils/data_config.yaml'
TRAINER_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/trainer/bert_train_config.yaml'
MODEL_CONFIG_PATH = 'Ebiose_CLGP/ebiose_clgp/model/bert_model_config.yaml'

def load_model(config):
    if config.text_encoder.name == 'Transformer':
        config.graph_node_tokenizer_max_pos = get_max_position_embedding(config.graph_feature_tokenizer)
        config.prompt_tokenizer_max_pos = get_max_position_embedding(config.prompt_tokenizer)
        tokenizer = None
        model = None
    elif config.text_encoder.name == 'Bert':
        tokenizer, model = get_Bert()
        config.embed_dim = 768  # Bert embedding dimension
    else:
        raise ValueError("Unsupported text encoder type.")
    
    model = CLGP(config, model)
    return model, tokenizer

def load_dataset(config, tokenizer):
    dataset = CLGP_Ebiose_dataset(config, tokenizer=tokenizer)
    return dataset

def load_checkpoint(config, model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if config.n_gpu > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def prepare_dataloader(config, dataset):
    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, num_workers=config.num_workers, collate_fn=collate_graph)
    return dataloader

def inference(model, dataloader, config):
    model.to(torch.device(config.device))
    model.eval()
    
    results = []
    with torch.no_grad():
        for batch in dataloader:
            graphs, texts, labels = batch
            graphs = graphs.to(torch.device(config.device))
            texts = texts.to(torch.device(config.device))

            graph_features, text_features = model(graphs, texts)
            similarities = torch.cosine_similarity(graph_features, text_features)
            results.extend(similarities.cpu().numpy())
    
    return results

def main():
    data_config = load_config_file(DATA_CONFIG_PATH)
    train_config = load_config_file(TRAINER_CONFIG_PATH)
    model_config = load_config_file(MODEL_CONFIG_PATH)
    config = OmegaConf.merge(train_config, data_config, model_config)
    
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count()
    
    model, tokenizer = load_model(config)
    dataset = load_dataset(config, tokenizer)
    test_dataset = dataset.train_validation_test_split()[-1]  # Assuming you want the test dataset

    model = load_checkpoint(config, model, config.saved_model)
    
    test_dataloader = prepare_dataloader(config, test_dataset)
    
    results = inference(model, test_dataloader, config)
    print("Inference Results:", results)
    
    wandb.finish()

if __name__ == "__main__":
    main()
