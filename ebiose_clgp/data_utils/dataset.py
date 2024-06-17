from torch.utils.data import Dataset
import torch
import pickle as pkl
import json
from tokenizers import Tokenizer
from torch_geometric.data import Data

class CLGP_Ebiose_dataset(Dataset):
    """CLGP_Ebiose_dataset. To train CLGP on prompt-graphs pairs."""

    def __init__(self, config, tokenizer = None):
        super(CLGP_Ebiose_dataset, self).__init__()

        if tokenizer == None:
            self.custom_tokenizer = True
            self.config = config
            self.prompt_context_length = self.config.prompt_context_length
            self.node_feature_context_length = self.config.node_feature_context_length
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

        self.pairs = self.create_pairs()

    def create_pairs(self):
        pairs = []
        for graph, evaluations in self.graph_data:
            for i in range(len(evaluations['evaluations'])):
                if evaluations['evaluations'][i]:  # Only consider successful evaluations
                    prompt = self.prompts_data[evaluations['dataset_indexes'][i]]
                    pairs.append((graph['graph'], prompt))
        return pairs

    def tokenize_prompt(self, text):
        if self.custom_tokenizer:
            tokens = self.prompt_tokenizer.encode(text).ids
            result = torch.zeros(self.prompt_context_length, dtype=torch.long)
            result[:len(tokens)] = torch.tensor(tokens[:self.prompt_context_length])  # Truncate if necessary
            return result
        else:
            return self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        graph, question = self.pairs[idx]
        graph_input = self.process_graph(graph)
        text_input = self.tokenize_prompt(question)
        return graph_input, text_input
    
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
        
        if self.custom_tokenizer:
            for features in node_features:
                tokenized_features = torch.zeros((self.node_feature_context_length), dtype=torch.long)
                tokens = self.graph_feature_tokenizer.encode(features).ids
                tokenized_features[:len(tokens)] = torch.tensor(tokens)
                node_features_tensor.append(tokenized_features)
                
        else:
            for features in node_features:
                node_features_tensor.append(self.tokenizer(features, return_tensors='pt', padding=True, truncation=True))
                
        node_features_tensor = torch.stack(node_features_tensor).float()  # Ensure node features are float

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return (node_features_tensor, edge_index)
