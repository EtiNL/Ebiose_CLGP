import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ebiose_clgp.model.graph_encoders.graph_convolutionnal_network import GCN
from ebiose_clgp.model.text_encoders.transformer import Transformer

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
                self.text_encoder = Transformer(config, 'prompt')
            except Exception as e: 
                raise Exception(f"Problem while instantiating the text_encoder model: {e}")
        else:
            raise Exception('Invalid config.text_encoder.name')

        if self.config.node_feature_encoder.name == 'Transformer':
            try:
                self.node_feature_encoder = Transformer(config, 'node_feature')
            except Exception as e:
                raise Exception(f"Problem while instantiating the node_feature_encoder model: {e}")
        else:
            raise Exception('Invalid config.node_feature_encoder.name')
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        self.graph_encoder.initialize()
        self.text_encoder.initialize()
        self.node_feature_encoder.initialize()

    def forward(self, graphs, texts):
        node_features = graphs.x
        embedded_node_features = self.node_feature_encoder(node_features.long())
        print('embedded_node_features.shape :', embedded_node_features.shape)
        graphs.x = embedded_node_features

        graph_features = self.graph_encoder(graphs)
        text_features = self.text_encoder(texts)

        return graph_features, text_features