from torch.utils.data import Dataset
import torch
import pickle as pkl
import json
import numpy as np
from torch_geometric.data import Data
    
class CLGP_Ebiose_dataset(Dataset):
    """CLGP_Ebiose_dataset. To train CLGP on prompt-graphs pairs."""

    def __init__(self, config, prompt_tokenizer, graph_feature_tokenizer, context_length=77, input_resolution=224):
        super(CLGP_Ebiose_dataset, self).__init__()

        self.config = config
        self.context_length = context_length
        self.prompt_tokenizer = prompt_tokenizer
        self.graph_feature_tokenizer = graph_feature_tokenizer

        # Load annotations and validation set
        with open(self.config.train_annotation_file, 'r') as f:
            self.annotations = [json.loads(line) for line in f]

        validation_set = pkl.load(open("data/gsm8k-validation.pkl", "rb"))
        self.validation_questions = validation_set["question"]
        self.validation_targets = validation_set["target"]

        self.pairs = self.create_pairs()

    def create_pairs(self):
        pairs = []
        for annotation in self.annotations:
            evaluations = annotation[1]["evaluations"]
            for idx, eval_result in enumerate(evaluations):
                if eval_result:  # Only consider successful evaluations
                    question = self.validation_questions[annotation[1]["dataset_indexes"][idx]]
                    graph = annotation[0]["graph"]
                    pairs.append((graph, question))
        return pairs

    def tokenize_prompt(self, text):
        tokens = self.prompt_tokenizer.encode(text).ids
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        graph, question, target = self.pairs[idx]
        graph_input = self.process_graph(graph)
        text_input = self.tokenize_prompt(question)
        return graph_input, text_input

    def process_graph(self, graph):
        graph_data = graph_struct_to_tokenized_graph_data(graph)
        
        encoder_name = self.config.graph_encoder
        
        if encoder_name == "graph_sage":
            graph_inputs = graph_data.x, graph_data.edge_index, graph_data.batch
        elif encoder_name == "dgcnn":
            graph_inputs = graph_data.x, graph_data.batch
        elif encoder_name == "gin":
            graph_inputs = graph_data.x, graph_data.edge_index, graph_data.batch
        elif encoder_name == "gat":
            graph_inputs = graph_data.x, graph_data.edge_index, graph_data.batch
        elif encoder_name == "gcn":
            graph_inputs = graph_data.x, graph_data.edge_index, graph_data.batch

        return graph_inputs
    
    
def graph_struct_to_tokenized_graph_data(graph_struct):
    G = parse_graph(graph_struct)

    # Extract node features
    node_features = np.array([G.nodes[i].get('features', [0, 0, 0]) for i in G.nodes])  # Default to [0, 0, 0] if no features

    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(np.array(list(G.edges)).T, dtype=torch.long)

    # Create a PyTorch Geometric data object
    graph_data = Data(x=x, edge_index=edge_index)
    
    return graph_data

def parse_graph(graph_struct):
    # Extract node features
    nodes = graph_struct.get("nodes", [])
    edges = graph_struct.get("edges", [])

    node_features = []
    node_id_map = {}
    for i, node in enumerate(nodes):
        node_id_map[node["id"]] = i
        node_features.append(node.get('features', [0, 0, 0]))  # Default to [0, 0, 0] if no features

    # Convert node features to tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Create edge index
    edge_index = []
    for edge in edges:
        start_node = node_id_map[edge["start_node_id"]]
        end_node = node_id_map[edge["end_node_id"]]
        edge_index.append([start_node, end_node])

    # Convert edge index to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create a PyTorch Geometric data object
    graph_data = Data(x=x, edge_index=edge_index)
    
    return graph_data