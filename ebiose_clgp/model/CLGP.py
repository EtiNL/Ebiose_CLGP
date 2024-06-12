# adapted code from https://github.com/ngthanhtin/CLIP-Training-Pytorch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ebiose_clgp.model.graph_encoders.graph_convolutionnal_network import GCN
from ebiose_clgp.model.text_encoders.transformer import Transformer


class CLGP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 config: str,
                 context_length: int,
                 vocab_size: int):
        
        super().__init__()

        self.context_length = context_length
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.config = config

        if self.config.graph_encoder.name == 'GCN':
            self.graph_encoder = GCN(self.config.node_feature_context_length, self.config.graph_encoder.hidden, self.embed_dim)
        else:
            raise Exception('invalid config.graph_encoder.name')
        
        if self.config.text_encoder.name == 'Transformer':
            self.text_encoder = Transformer(self.config.prompt_context_length, self.config.text_encoder.width, self.config.text_encoder.layers, self.config.text_encoder.heads)
        else:
            raise Exception('invalid config.text_encoder.name')
        
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