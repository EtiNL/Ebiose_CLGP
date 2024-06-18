from torch.utils.data import Dataset, random_split, Subset
import torch
import pickle as pkl
import json
from tokenizers import Tokenizer
import hashlib
from tqdm import tqdm

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

        with open(self.config.graph_data_file, 'r') as f:
            self.graph_data = []
            for line in f:
                self.graph_data.append(json.loads(line))

        validation_set = pkl.load(open(self.config.gsm8k_validation_file, "rb"))
        self.prompts_data = validation_set["question"]

        self.graph_hashmap = {}
        self.prompt_hashmap = {}
        self.evaluation_map = {}
        
        self.index_map = {}  # Dictionary to map index to (graph_hash, prompt_hash)
        self.pairs = self.create_pairs()

    def create_pairs(self):
        print("creating pairs...")
        pairs = []
        for idx, (graph, evaluations) in enumerate(tqdm(self.graph_data)):
            for i in range(len(evaluations['evaluations'])):
                processed_graph = self.process_graph(graph['graph'])
                node_features_tensor, edge_index = processed_graph
                prompt = self.prompts_data[evaluations['dataset_indexes'][i]]
                tokenized_prompt = self.tokenize_prompt(prompt)

                graph_hash = self.hash_tensor(node_features_tensor)
                prompt_hash = self.hash_tensor(tokenized_prompt)

                self.graph_hashmap[graph_hash] = processed_graph
                self.prompt_hashmap[prompt_hash] = tokenized_prompt
                self.evaluation_map[(graph_hash, prompt_hash)] = evaluations['evaluations'][i]

                if evaluations['evaluations'][i]:  # Only consider successful evaluations
                    pairs.append((processed_graph, tokenized_prompt))
                    self.index_map[len(pairs) - 1] = (graph_hash, prompt_hash)
        print("end creating pairs")
        return pairs

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
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

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

    def train_validation_test_split(self, train_ratio=0.8, val_ratio=0.1):
        print("begin split...")
        train_size = int(train_ratio * len(self))
        val_size = int(val_ratio * len(self))
        remaining_size = len(self) - train_size - val_size
        
        # Perform initial random split to get train, val and remaining sets
        train_val_indices, test_indices = random_split(range(len(self)), [train_size + val_size, remaining_size])
        train_indices, val_indices = random_split(train_val_indices, [train_size, val_size])
        
        # Get tokenized prompts for train and validation sets
        train_prompts = {self.pairs[idx][1].numpy().tobytes() for idx in train_indices}
        val_prompts = {self.pairs[idx][1].numpy().tobytes() for idx in val_indices}
        
        # Combine train and validation prompts
        seen_prompts = train_prompts.union(val_prompts)
        
        # Filter out pairs in test set with tokenized prompts already in train or val set
        filtered_test_indices = [idx for idx in test_indices if self.pairs[idx][1].numpy().tobytes() not in seen_prompts]
        
        assert len(filtered_test_indices) == 0, "test dataset is empty"
        
        return Subset(self, train_indices), Subset(self, val_indices), Subset(self, filtered_test_indices)
