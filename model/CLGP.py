# adapted code from https://github.com/ngthanhtin/CLIP-Training-Pytorch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CLGP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # graph
                 Graph_encoder,
                 graph_encoder_config,
                 # text
                 Text_encoder,
                 text_encoder_config,
                 context_length: int,
                 vocab_size: int):
        
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.graph_encoder = Graph_encoder(graph_encoder_config)
        self.text_encoder = Text_encoder(text_encoder_config)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):

        #initialise graph_encoder
        self.graph_encoder.initialize(self.embed_dim)

        #initialise text_encoder
        self.text_encoder.initialize(self.embed_dim, self.vocab_size)


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, graph, text):
        graph_features = self.graph_encoder.encode_graph(graph)
        text_features = self.text_encoder.encode_text(text)

        return graph_features, text_features 