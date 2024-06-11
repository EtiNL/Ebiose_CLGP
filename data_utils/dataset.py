from torch.utils.data import Dataset
import torch
import pickle as pkl
import json
from tokenizers import Tokenizer
from torch_geometric.data import Data
    
class CLGP_Ebiose_dataset(Dataset):
    """CLGP_Ebiose_dataset. To train CLGP on prompt-graphs pairs."""

    def __init__(self, config_data, graph_encoder_name):
        super(CLGP_Ebiose_dataset, self).__init__()

        self.config = config_data
        self.prompt_context_length = self.config.prompt_context_length
        self.node_feature_context_length = self.config.node_feature_context_length
        self.prompt_tokenizer = Tokenizer.from_file(self.config.prompt_tokenizer)
        self.graph_feature_tokenizer = Tokenizer.from_file(self.config.graph_feature_tokenizer)
        self.graph_encoder_name = graph_encoder_name
        
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
        tokens = self.prompt_tokenizer.encode(text).ids
        result = torch.zeros(self.prompt_context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        graph, question = self.pairs[idx]
        graph_input = self.process_graph(graph)
        text_input = self.tokenize_prompt(question)
        return graph_input, text_input

    def process_graph(self, graph):
        graph_data = self.parse_graph(graph)
        
        if self.graph_encoder_name == "graph_sage":
            graph_inputs = graph_data.x, graph_data.edge_index, graph_data.batch
        elif self.graph_encoder_name == "dgcnn":
            graph_inputs = graph_data.x, graph_data.batch
        elif self.graph_encoder_name == "gin":
            graph_inputs = graph_data.x, graph_data.edge_index, graph_data.batch
        elif self.graph_encoder_name == "gat":
            graph_inputs = graph_data.x, graph_data.edge_index, graph_data.batch
        elif self.graph_encoder_name == "gcn":
            graph_inputs = graph_data.x, graph_data.edge_index, graph_data.batch

        return graph_inputs
    
    def parse_graph(self, graph_struct):
        # Extract node features
        shared_context_prompt = graph_struct.get('shared_context_prompt','')
        nodes = graph_struct.get("nodes", [])
        edges = graph_struct.get("edges", [])

        node_features = []
        node_id_map = {}
        for i, node in enumerate(nodes):
            node_id_map[node["id"]] = i
            node_features.append(['name:' + node.get('id', '') + '    purpose: ' + node.get('purpose', '') + '     type: ' + node.get('type','')+ '   model: ' + node.get('model','')+ '     shared_context_prompt: ' + shared_context_prompt])

        # Create edge index
        edge_index = []
        for edge in edges:
            start_node = node_id_map[edge["start_node_id"]]
            end_node = node_id_map[edge["end_node_id"]]
            if edge.get('condition','') != '':
                condition_node = ['name:' + edge['condition']+ '    purpose: ' + 'Allows acces to the next step if verified'+ '     type: ' + 'condition'+ '    model: ', '     shared_context_prompt: ']
                node_features.append(condition_node)
                edge_index.append([start_node, len(node_features)-1])
                edge_index.append([len(node_features)-1, end_node])
            else:
                edge_index.append([start_node, end_node])

        # Tokenize Node Feature
        node_features = []
        for features in node_features:
            tokenized_features = torch.zeros((self.node_feature_context_length), dtype=torch.long)
            tokens = self.graph_feature_tokenizer.encode(features).ids
            tokenized_features[:len(tokens)] = tokens 
            node_features.append(tokenized_features)
        node_features = torch.stack(node_features)
        
        # Convert node features to tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create a PyTorch Geometric data object
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        return graph_data

