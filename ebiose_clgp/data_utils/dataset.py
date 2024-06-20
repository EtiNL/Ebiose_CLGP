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

class CLGP_Ebiose_dataset(Dataset):
    """CLGP_Ebiose_dataset. To train CLGP on prompt-graphs pairs."""

    def __init__(self, config, tokenizer=None):
        super(CLGP_Ebiose_dataset, self).__init__()

        self.config = config
        self.prompt_context_length = self.config.prompt_context_length
        self.node_feature_context_length = self.config.node_feature_context_length
        
        if tokenizer is None:
            self.custom_tokenizer = True
            self.prompt_tokenizer = Tokenizer.from_file(self.config.prompt_tokenizer)
            self.graph_feature_tokenizer = Tokenizer.from_file(self.config.graph_feature_tokenizer)
        else:
            self.custom_tokenizer = False
            self.tokenizer = tokenizer

        # Check if the dataset file is zipped and unzip it
        dataset_file_path = self.config.dataset_file
        
        if os.path.exists(dataset_file_path):
            if dataset_file_path.endswith('.zip'):
                print('unzipping dataset...')
                with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(dataset_file_path))
                dataset_file_path = dataset_file_path.rstrip('.zip')+'.pkl' # Update the path to the unzipped file
                print("done")
            print("loading dataset...")
            self.data, self.index_map = self.load_data_and_index_map(dataset_file_path)
            print("done")
        else:
            print(f"dataset_file_path: {dataset_file_path} doesn't exist")
            with open(self.config.graph_data_file, 'r') as f:
                self.graph_data = [json.loads(line) for line in f]

            validation_set = pkl.load(open(self.config.gsm8k_validation_file, "rb"))
            self.prompts_data = validation_set["question"]
            self.index_map = {}
            self.data = self.create_data_and_index_map()
            self.save_pairs_and_maps(dataset_file_path)

    def create_data_and_index_map(self):
        print("creating pairs...")
        data = []
        for idx, (graph, evaluations) in enumerate(tqdm(self.graph_data)):
            for i in range(len(evaluations['evaluations'])):
                processed_graph = self.process_graph(graph['graph'])
                node_features_tensor, edge_index = processed_graph
                prompt = self.prompts_data[evaluations['dataset_indexes'][i]]
                tokenized_prompt = self.tokenize_prompt(prompt)

                graph_hash = self.hash_tensor(node_features_tensor)
                prompt_hash = self.hash_tensor(tokenized_prompt)
                
                pair_eval = torch.tensor([1]) if evaluations['evaluations'][i] else torch.tensor([0])
                
                data.append((processed_graph, tokenized_prompt, pair_eval))
                self.index_map[len(data) - 1] = (graph_hash, prompt_hash, pair_eval)
                
        print("end creating pairs")
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

    def save_pairs_and_maps(self, file_path):
        """Save the pairs and related maps to a pickle file."""
        data = {
            'data': self.data,
            'index_map': self.index_map
        }
        with open(file_path, 'wb') as f:
            pkl.dump(data, f)

    def load_data_and_index_map(self, file_path):
        """Load the pairs and related maps from a pickle file."""
        with open(file_path, 'rb') as f:
            data = pkl.load(f)
        return data['data'], data['index_map']

    def train_validation_test_split(self, num_isolated_prompts = 10, num_isolated_graphs = 10, train_ratio=0.8, val_ratio=0.2):
        total_pairs = len(self.data)
        all_indices = list(range(total_pairs))
        
        isolated_prompt_indices = random.sample(all_indices, num_isolated_prompts)
        remaining_indices = list(set(all_indices) - set(isolated_prompt_indices))
        indices_to_transfer = []
        
        for id_x in isolated_prompt_indices:
            graph_id, prompt_id, _ = self.index_map[id_x]
            for id_y in remaining_indices:
                if prompt_id == self.index_map[id_y][1]:
                    indices_to_transfer.append(id_y)
                    
        isolated_prompt_indices = list(set(isolated_prompt_indices) | set(indices_to_transfer))
        remaining_indices = list(set(remaining_indices) - set(indices_to_transfer))

        isolated_graph_indices = random.sample(isolated_prompt_indices, num_isolated_graphs)
        indices_to_transfer = []
        
        for id_x in isolated_graph_indices:
            graph_id, prompt_id, _ = self.index_map[id_x]
            for id_y in remaining_indices:
                if graph_id == self.index_map[id_y][0]:
                    indices_to_transfer.append(id_y)
                    
        test_indices = list(set(isolated_prompt_indices) | set(indices_to_transfer))
        remaining_indices = list(set(remaining_indices) - set(indices_to_transfer))
        
        num_train = int(train_ratio * len(remaining_indices))
        num_val = int(val_ratio * len(remaining_indices))
        
        random.shuffle(remaining_indices)
        train_indices = remaining_indices[:num_train]
        val_indices = remaining_indices[num_train:num_train + num_val]
        
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        cat_1, cat_2, cat_3 = self.categorize_test_indices()
        
        wandb.config["num_isolated_prompts"] = num_isolated_prompts
        wandb.config["num_isolated_graphs"] = num_isolated_graphs
        wandb.config["num test cat_1"] = len(cat_1)
        wandb.config["num test cat_2"] = len(cat_2)
        wandb.config["num test cat_3"] = len(cat_3)
        
        print("test categories populations:", len(cat_1), len(cat_2), len(cat_3))
        
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset_cat_1 = Subset(self, cat_1)
        test_dataset_cat_2 = Subset(self, cat_2)
        test_dataset_cat_3 = Subset(self, cat_3)
        
        return train_dataset, val_dataset, test_dataset_cat_1, test_dataset_cat_2, test_dataset_cat_3
    
    def categorize_test_indices(self):
        train_val_indices = self.train_indices + self.val_indices
        
        train_val_graphs = set()
        train_val_prompts = set()
        
        for idx in train_val_indices:
            graph_id, prompt_id, _ = self.index_map[idx]
            train_val_graphs.add(graph_id)
            train_val_prompts.add(prompt_id)
        
        category_1 = [] # graph_id not in train_val_graphs and prompt_id in train_val_prompts
        category_2 = [] # prompt_id not in train_val_prompts and graph_id in train_val_graphs
        category_3 = [] # graph_id not in train_val_graphs and prompt_id not in train_val_prompts
        
        for idx in self.test_indices:
            graph_id, prompt_id, _ = self.index_map[idx]
            if graph_id not in train_val_graphs and prompt_id in train_val_prompts:
                category_1.append(idx)
            elif prompt_id not in train_val_prompts and graph_id in train_val_graphs:
                category_2.append(idx)
            elif graph_id not in train_val_graphs and prompt_id not in train_val_prompts:
                category_3.append(idx)
        
        return category_1, category_2, category_3
