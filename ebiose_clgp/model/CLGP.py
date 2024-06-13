import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ebiose_clgp.model.graph_encoders.graph_convolutionnal_network import GCN
from ebiose_clgp.model.text_encoders.transformer import Transformer
from ebiose_clgp.model.graph_encoders.graph_utils import combine_graphs

class CLGP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 config: dict,
                 context_length: int,
                 vocab_size: int):
        
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.config = config

        if self.config.graph_encoder.name == 'GCN':
            try:
                self.graph_encoder = GCN(self.config.node_feature_context_length, self.config.graph_encoder.hidden, self.embed_dim)
            except Exception as e: 
                raise Exception(f"Problem while instantiating the graph_encoder model: {e}")
        else:
            raise Exception('Invalid config.graph_encoder.name')
        
        if self.config.text_encoder.name == 'Transformer':
            try:
                self.text_encoder = Transformer(self.config.prompt_context_length, self.config.text_encoder.width, self.config.text_encoder.layers, self.config.text_encoder.heads)
            except Exception as e: 
                raise Exception(f"Problem while instantiating the text_encoder model: {e}")
        else:
            raise Exception('Invalid config.text_encoder.name')
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        self.graph_encoder.initialize(self.embed_dim)
        self.text_encoder.initialize(self.embed_dim, self.vocab_size)

    def forward(self, graphs, texts):
        combined_graph = combine_graphs(graphs)
        graph_features = self.graph_encoder(combined_graph)
        text_features = self.text_encoder(texts)

        return graph_features, text_features
