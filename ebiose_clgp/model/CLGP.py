import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ebiose_clgp.model.graph_encoders.graph_convolutionnal_network import GCN
from ebiose_clgp.model.text_encoders.transformer import Transformer
from ebiose_clgp.model.graph_encoders.graph_utils import combine_graphs

class CLGP(nn.Module):
    def __init__(self, config: dict):
        
        super().__init__()
        
        self.config = config

        if self.config.graph_encoder.name == 'GCN':
            try:
                self.graph_encoder = GCN(config)
            except Exception as e: 
                raise Exception(f"Problem while instantiating the graph_encoder model: {e}")
        else:
            raise Exception('Invalid config.graph_encoder.name')
        
        if self.config.text_encoder.name == 'Transformer':
            try:
                self.text_encoder = Transformer(config)
            except Exception as e: 
                raise Exception(f"Problem while instantiating the text_encoder model: {e}")
        else:
            raise Exception('Invalid config.text_encoder.name')
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        self.graph_encoder.initialize()
        self.text_encoder.initialize()

    def forward(self, graphs, texts):
        # print(graphs.type())
        combined_graph = combine_graphs(graphs)
        graph_features = self.graph_encoder(combined_graph)
        text_features = self.text_encoder(texts)

        return graph_features, text_features
