# dataloader here
from torch.utils.data import Dataset

from omegaconf import OmegaConf
import os.path as op
import random

from ..utils import load_from_yaml_file, read_json, load_config_file


import torch
import numpy as np

def _transform(n_px):
    #feature matrix normalization for graphs, 
    return None

class CLGP_Ebiose_dataset(Dataset):
    """CLGP_Ebiose_dataset. To train CLGP on prompt-graphs pairs."""

    def __init__(self, config, text_tokenizer, context_length=77, input_resolution=224):
        
        super(CLGP_Ebiose_dataset, self).__init__()

        self.config = config

        annotation_file = self.config.train_annotation_file
        # print("annotation_file : ", annotation_file)
        annotations = read_json(annotation_file)
        self.transform = _transform(input_resolution)
        self._tokenizer = text_tokenizer
        self.context_length = context_length


    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        graph_input = None
        
        text = None
        text_input = self.tokenize(text)

        return graph_input, text_input