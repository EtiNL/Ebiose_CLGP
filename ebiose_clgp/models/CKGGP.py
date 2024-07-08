
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



class CKGGP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text_to_knowledge_graph
                 text_to_kg_model,
                 # graph
                 Graph_encoder,
                 kg_encoder_config,
                 graph_encoder_config,
                 ):
        
        super().__init__()

        self.text_to_kg_model = text_to_kg_model
        self.embed_dim = embed_dim
        
        self.kg_encoder = Graph_encoder(kg_encoder_config)
        self.graph_encoder = Graph_encoder(graph_encoder_config)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):

        #initialise graph_encoder
        self.graph_encoder.initialize(self.embed_dim)

        #initialise kg_encoder
        self.kg_encoder.initialize(self.embed_dim)


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, graph, text):
        graph_features = self.graph_encoder.encode_graph(graph)
        kg_features = self.kg_encoder.encode_graph(self.text_to_kg_model(text))

        return graph_features, kg_features 