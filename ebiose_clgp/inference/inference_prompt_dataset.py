from torch.utils.data import Dataset, random_split, Subset
import torch
import pickle as pkl
import json
from tokenizers import Tokenizer
import hashlib
from tqdm import tqdm
import random
import numpy as np
import os
import wandb
import zipfile

class Inference_prompt_dataset(Dataset):

    def __init__(self, config, tokenizer=None):
        super(Inference_prompt_dataset, self).__init__()

        self.config = config
        self.prompt_context_length = self.config.prompt_context_length
        
        if tokenizer is None:
            self.custom_tokenizer = True
            self.prompt_tokenizer = Tokenizer.from_file(self.config.prompt_tokenizer)
            self.graph_feature_tokenizer = Tokenizer.from_file(self.config.graph_feature_tokenizer)
        else:
            self.custom_tokenizer = False
            self.tokenizer = tokenizer

        # Check if the dataset file is zipped and unzip it
        inference_prompt_dataset_file_path = self.config.inference_prompt_dataset_file
        
        if os.path.exists(inference_prompt_dataset_file_path):
            if inference_prompt_dataset_file_path.endswith('.zip'):
                print('unzipping prompt inference dataset...')
                with zipfile.ZipFile(inference_prompt_dataset_file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(inference_prompt_dataset_file_path))
                dataset_file_path = inference_prompt_dataset_file_path.rstrip('.zip')+'.pkl' # Update the path to the unzipped file
                print("done")
            print("loading prompt inference dataset...")
            self.prompt_data = pkl.load(open(dataset_file_path, "rb"))
            self.data = self.create_data()
            print("done")
        else:
            raise Exception(f"{dataset_file_path} not found")

    def create_data(self):
        data = []

        for prompt in self.prompt_data:
            tokenized_prompt = self.tokenize_prompt(prompt)
            data.append(tokenized_prompt)
                
        return data

    def tokenize_prompt(self, text):
        max_length = self.prompt_context_length
        if self.custom_tokenizer:
            tokens = self.prompt_tokenizer.encode(text).ids
            result = torch.zeros(self.prompt_context_length, dtype=torch.long)
            result[:len(tokens)] = torch.tensor(tokens[:self.prompt_context_length])  # Truncate if necessary
            return result
        else:
            tokenized_features = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids'][0]
            if tokenized_features.size(0) < max_length:
                padded_features = torch.cat((tokenized_features, torch.zeros(max_length - tokenized_features.size(0), dtype=torch.long)))
            else:
                padded_features = tokenized_features[:max_length]
                
            return padded_features

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def hash_tensor(self, tensor):
        """Generate a hash for a given tensor."""
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
