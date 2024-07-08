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

class Inferance_graph_dataset(Dataset):

    def __init__(self, config, tokenizer=None):
        super(Inferance_graph_dataset, self).__init__()

        self.config = config
        self.node_feature_context_length = self.config.node_feature_context_length
        
        if tokenizer is None:
            self.custom_tokenizer = True
            self.graph_feature_tokenizer = Tokenizer.from_file(self.config.graph_feature_tokenizer)
        else:
            self.custom_tokenizer = False
            self.tokenizer = tokenizer

        # Check if the dataset file is zipped and unzip it
        inference_graph_dataset_file_path = self.config.dataset_file
        
        if os.path.exists(inference_graph_dataset_file_path):
            if inference_graph_dataset_file_path.endswith('.zip'):
                print('unzipping dataset...')
                with zipfile.ZipFile(inference_graph_dataset_file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(inference_graph_dataset_file_path))
                dataset_file_path = inference_graph_dataset_file_path.rstrip('.zip')+'.pkl' # Update the path to the unzipped file
                print("done")
            print("loading inference graph dataset...")
            with open(dataset_file_path, 'r') as f:
                self.graph_data = [json.loads(line) for line in f]
            self.graph_id_map = {}
            self.data = self.create_data()
            print("done")
        else:
            raise Exception(f"{dataset_file_path} not found")
            

    def create_data(self):
        data = []
        for graph, evaluation in enumerate(self.graph_data):
            processed_graph = self.process_graph(graph['graph'])
            node_features_tensor, edge_index = processed_graph

            graph_id = graph['id']
            graph_hash = self.hash_tensor(node_features_tensor)
            
            data.append(processed_graph)
            self.graph_id_map[graph_hash] = graph_id
            
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def process_graph(self, graph_struct):
        # Extract node features
        shared_context_prompt = graph_struct.get('shared_context_prompt', '')
        nodes = graph_struct.get("nodes", [])
        edges = graph_struct.get("edges", [])

        node_features = []
        node_id_map = {}
        for i, node in enumerate(nodes):
            node_id_map[node["id"]] = i
            node_features.append('name:' + node.get('id', '') + '    purpose: ' + node.get('purpose', '') + '     type: ' + node.get('type', '') + '   model: ' + node.get('model', '') + '     shared_context_prompt: ' + shared_context_prompt)

        # Create edge index
        edge_index = []
        for edge in edges:
            start_node = node_id_map[edge["start_node_id"]]
            end_node = node_id_map[edge["end_node_id"]]
            if edge.get('condition', '') != '':
                condition_node = 'name:' + edge['condition'] + '    purpose: ' + 'Allows access to the next step if verified' + '     type: ' + 'condition' + '    model: ' + '     shared_context_prompt: '
                node_features.append(condition_node)
                edge_index.append([start_node, len(node_features) - 1])
                edge_index.append([len(node_features) - 1, end_node])
            else:
                edge_index.append([start_node, end_node])

        # Tokenize Node Features
        node_features_tensor = []
        max_length = self.node_feature_context_length
        
        if self.custom_tokenizer:
            for features in node_features:
                tokenized_features = torch.zeros((max_length), dtype=torch.long)
                tokens = self.graph_feature_tokenizer.encode(features).ids
                tokenized_features[:min(len(tokens), max_length)] = torch.tensor(tokens[:max_length])
                node_features_tensor.append(tokenized_features)
                
        else:
            for features in node_features:
                tokenized_features = self.tokenizer(features, return_tensors='pt', padding=True, truncation=True)['input_ids'][0]
                if tokenized_features.size(0) < max_length:
                    padded_features = torch.cat((tokenized_features, torch.zeros(max_length - tokenized_features.size(0), dtype=torch.long)))
                else:
                    padded_features = tokenized_features[:max_length]
                node_features_tensor.append(padded_features)
                
        node_features_tensor = torch.stack(node_features_tensor).float()  # Ensure node features are float

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return (node_features_tensor, edge_index)

    def hash_tensor(self, tensor):
        """Generate a hash for a given tensor."""
        return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

